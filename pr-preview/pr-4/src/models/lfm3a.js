// LFM3 audio encoder (from the Omni mmproj): log-mel [128 × frames] → three stride-2 conv blocks (8× shorter in
// time) → Linear 4096→768 → 17 causal layers (width-3 short conv or sliding-window GQA attention) → RMSNorm →
// LayerNorm + GELU MLP into the language model's embedding space. One audio token per 8 mel frames (80 ms).
// Loaded on demand and freed after use, like the vision encoder.

import { streamTensors, freeTensors } from '../loader.js';
import * as K from '../kernels.js';

const QB = 64;
const SRC = {};
const src = (name, mt, f) => (SRC[name + mt] ||= f(mt));

export class LFM3Audio {
  /** @param {import('../gpu.js').Gpu} gpu */
  static async load(gpu, source, gguf, { onProgress } = {}) {
    const tensors = gguf.tensors.filter((t) => /^a\.(conv1d|pre_encode|blk|post_ln)/.test(t.name) || t.name.startsWith('mm.a.'));
    if (!tensors.length) throw new Error('this mmproj has no audio encoder');
    const a = new LFM3Audio(gpu, gguf.kv);
    a.w = await streamTensors(gpu, source, tensors, { tag: 'audio', onProgress });
    return a;
  }

  constructor(gpu, kv) {
    this.gpu = gpu;
    this.C = kv['clip.audio.embedding_length'];
    this.F = kv['clip.audio.feed_forward_length'];
    this.nLayer = kv['clip.audio.block_count'];
    this.H = kv['clip.audio.attention.head_count'];
    this.HK = kv['clip.audio.attention.head_count_kv'];
    this.D = this.C / this.H;
    this.eps = kv['clip.audio.attention.layer_norm_epsilon'];
    this.window = kv['clip.audio.window_size'];
    this.out = kv['clip.audio.projection_dim'];
    this.ropeBase = 1e6;
  }

  #u(vals, floats = []) {
    const a = new Uint32Array(vals.length), f = new Float32Array(a.buffer);
    vals.forEach((v, i) => { if (floats.includes(i)) f[i] = v; else a[i] = v; });
    return a;
  }
  #ew(n) { return [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)]; }
  #mm(enc, name, X, Y, T, bias) {
    const w = this.w[name], mt = K.MAXT;
    const groups = [Math.min(w.rows, 32768), Math.ceil(w.rows / 32768), Math.ceil(T / mt)];
    const b = bias ? this.w[bias].buf : this.zero;
    if (w.type === 'Q8_0') this.gpu.dispatch(enc, src('q8', mt, K.matmulQ8_0), [this.#u([w.rows, w.K, T, w.dOff, bias ? 1 : 0]), w.buf, X, Y, b], groups);
    else this.gpu.dispatch(enc, src('f32', mt, K.matmulF32), [this.#u([w.rows, w.K, T, bias ? 1 : 0]), w.buf, X, Y, b], groups);
  }
  #norm(enc, X, wname, Y, rows, N) {
    this.gpu.dispatch(enc, K.rmsnorm, [this.#u([N, 0, this.eps, N, 0, 0, 0, 0], [2]), X, this.w[wname].buf, Y], rows);
  }

  /**
   * Encodes log-mel features (audio.js `logMel`) into language-model embeddings.
   * @returns {Promise<{ buf: GPUBuffer, tokens: number }>} buf holds [tokens, out] floats; free it with gpu.free.
   */
  async encode({ mel, frames }) {
    const gpu = this.gpu, C = this.C, F = this.F, H = this.H, HK = this.HK, D = this.D, CH = 256;
    const down = (n) => Math.floor((n - 1) / 2) + 1;
    const T1 = down(frames), T2 = down(T1), N = down(T2);
    const F1 = 64, F2 = 32, F3 = 16;
    const f = (n) => gpu.buffer(n * 4, 'audio-act');
    const a = {
      mel: f(128 * frames), d0: f(CH * F1 * T1), d1: f(CH * F2 * T2), p1: f(CH * F2 * T2), d2: f(CH * F3 * N), p2: f(CH * F3 * N),
      flat: f(N * CH * F3), x: f(N * C), h: f(N * C), big: f(N * 3 * C), q: f(N * C), qn: f(N * C), k: f(N * HK * D),
      kn: f(N * HK * D), v: f(N * HK * D), att: f(N * C), y: f(N * C), g: f(N * F), u: f(N * F), S: f(QB * H * N),
      m1: f(N * this.out),
    };
    this.zero = f(4);
    const out = gpu.buffer(N * this.out * 4, 'audio-out');
    gpu.device.queue.writeBuffer(a.mel, 0, mel);
    const W = (n) => this.w[n].buf;
    try {
      let enc = gpu.device.createCommandEncoder();
      // Downsampler: conv(1→256) + SiLU, then twice depthwise conv + pointwise conv + SiLU.
      gpu.dispatch(enc, K.convDown, [this.#u([frames, 128, T1, F1, CH, 0, 1, 0]), a.mel, W('a.conv1d.0.weight'), W('a.conv1d.0.bias'), a.d0], [Math.ceil(T1 / 64), CH * F1]);
      gpu.dispatch(enc, K.convDown, [this.#u([T1, F1, T2, F2, CH, 1, 0, 0]), a.d0, W('a.conv1d.2.weight'), W('a.conv1d.2.bias'), a.d1], [Math.ceil(T2 / 64), CH * F2]);
      gpu.dispatch(enc, K.pointwise, [this.#u([CH, F2 * T2, 0, 0]), a.d1, W('a.conv1d.3.weight'), W('a.conv1d.3.bias'), a.p1], [Math.ceil((F2 * T2) / 64), CH]);
      gpu.dispatch(enc, K.convDown, [this.#u([T2, F2, N, F3, CH, 1, 0, 0]), a.p1, W('a.conv1d.5.weight'), W('a.conv1d.5.bias'), a.d2], [Math.ceil(N / 64), CH * F3]);
      gpu.dispatch(enc, K.pointwise, [this.#u([CH, F3 * N, 0, 0]), a.d2, W('a.conv1d.6.weight'), W('a.conv1d.6.bias'), a.p2], [Math.ceil((F3 * N) / 64), CH]);
      gpu.dispatch(enc, K.flattenCFT, [this.#u([CH, F3, N, 0]), a.p2, a.flat], N);
      this.#mm(enc, 'a.pre_encode.out.weight', a.flat, a.x, N, 'a.pre_encode.out.bias');
      await gpu.run(() => {}, enc);
      for (let i = 0; i < this.nLayer; i++) {
        enc = gpu.device.createCommandEncoder();
        const p = `a.blk.${i}.`;
        this.#norm(enc, a.x, p + 'ln1.weight', a.h, N, C);
        if (this.w[p + 'conv_dw.weight']) {
          this.#mm(enc, p + 'conv_pw1.weight', a.h, a.big, N);
          const width = this.w[p + 'conv_dw.weight'].K;
          gpu.dispatch(enc, src('sc', width, K.shortconvFresh), [this.#u([N, C, 0, 0]), a.big, W(p + 'conv_dw.weight'), a.att], [Math.ceil(C / 64), N]);
          this.#mm(enc, p + 'conv_pw2.weight', a.att, a.y, N);
        } else {
          this.#mm(enc, p + 'attn_q.weight', a.h, a.q, N);
          this.#mm(enc, p + 'attn_k.weight', a.h, a.k, N);
          this.#mm(enc, p + 'attn_v.weight', a.h, a.v, N);
          this.#norm(enc, a.q, p + 'attn_q_norm.weight', a.qn, N * H, D);
          this.#norm(enc, a.k, p + 'attn_k_norm.weight', a.kn, N * HK, D);
          gpu.dispatch(enc, K.ropeN, [this.#uf3(H, this.ropeBase, D), a.qn], N * H);
          gpu.dispatch(enc, K.ropeN, [this.#uf3(HK, this.ropeBase, D), a.kn], N * HK);
          for (let q0 = 0; q0 < N; q0 += QB) {
            const nq = Math.min(QB, N - q0);
            gpu.dispatch(enc, K.encScores, [this.#u([N, q0, H, HK, D, 1 / Math.sqrt(D), this.window, 0], [5]), a.qn, a.kn, a.S], [Math.ceil(N / 64), nq * H]);
            gpu.dispatch(enc, K.softmax, [this.#u([N, N, 0, 0]), a.S], nq * H);
            gpu.dispatch(enc, K.encOut, [this.#u([N, q0, H, HK, D, 0, 0, 0]), a.S, a.v, a.att], nq * H);
          }
          this.#mm(enc, p + 'attn_out.weight', a.att, a.y, N);
        }
        gpu.dispatch(enc, K.addInto, [this.#u([N * C]), a.y, a.x], this.#ew(N * C));
        this.#norm(enc, a.x, p + 'ln2.weight', a.h, N, C);
        this.#mm(enc, p + 'ffn_gate.weight', a.h, a.g, N);
        this.#mm(enc, p + 'ffn_up.weight', a.h, a.u, N);
        gpu.dispatch(enc, K.swiglu, [this.#u([N * F]), a.g, a.u], this.#ew(N * F));
        this.#mm(enc, p + 'ffn_down.weight', a.g, a.y, N);
        gpu.dispatch(enc, K.addInto, [this.#u([N * C]), a.y, a.x], this.#ew(N * C));
        await gpu.run(() => {}, enc);
      }
      enc = gpu.device.createCommandEncoder();
      this.#norm(enc, a.x, 'a.post_ln.weight', a.h, N, C);
      gpu.dispatch(enc, K.layernorm, [this.#u([C, 1e-5, 0, 0], [1]), a.h, W('mm.a.mlp.0.weight'), W('mm.a.mlp.0.bias'), a.y], N);
      this.#mm(enc, 'mm.a.mlp.1.weight', a.y, a.m1, N, 'mm.a.mlp.1.bias');
      gpu.dispatch(enc, K.geluErf, [this.#u([N * this.out]), a.m1], this.#ew(N * this.out));
      this.#mm(enc, 'mm.a.mlp.3.weight', a.m1, out, N, 'mm.a.mlp.3.bias');
      await gpu.run(() => {}, enc);
    } finally {
      for (const b of Object.values(a)) gpu.free(b);
      gpu.free(this.zero);
    }
    return { buf: out, tokens: N };
  }

  #uf3(H, base, D) { return this.#u([H, 0, base, D], [2]); }

  /** Frees the encoder's weights. */
  free() { freeTensors(this.gpu, this.w); this.w = {}; }
}
