// LFM3-VL vision encoder (from an Omni or VL mmproj): 16×16 patches → 14 layers mixing gated depthwise-conv
// blocks and 2D-RoPE attention → LayerNorm → 2×2 merge → MLP projector into the language model's embedding space.
// Loaded on demand and freed after use, so it only costs memory while an image is being read.

import { streamTensors, freeTensors } from '../loader.js';
import * as K from '../kernels.js';

const QB = 64; // queries per attention block (bounds the score buffer)
const SRC = {};
const src = (name, mt, f) => (SRC[name + mt] ||= f(mt));

export class LFM3Vision {
  /**
   * @param {import('../gpu.js').Gpu} gpu
   * @param {import('../gguf.js').ByteSource} source mmproj file
   * @param {Awaited<ReturnType<import('../gguf.js').readGGUF>>} gguf its parsed header
   */
  static async load(gpu, source, gguf, { onProgress } = {}) {
    const kv = gguf.kv;
    const tensors = gguf.tensors.filter((t) => t.name.startsWith('v.') || /^mm\.(0|2|input_norm)\./.test(t.name));
    if (!tensors.length) throw new Error('this mmproj has no vision encoder');
    const v = new LFM3Vision(gpu, kv);
    v.w = await streamTensors(gpu, source, tensors, { tag: 'vision', onProgress });
    // The patchifier has two temporal kernels: a video's frame pair goes through one each (W0·A + W1·B), and a still
    // image fills both slices, i.e. the same patches go through both.
    // GGUF dims (16, 16, 3, C): as a matmul each is C rows of 3·16·16 values.
    for (const n of ['v.patch_embd.weight', 'v.patch_embd.weight.1']) if (v.w[n]) { v.w[n].rows = v.C; v.w[n].K = 3 * 16 * 16; }
    return v;
  }

  constructor(gpu, kv) {
    this.gpu = gpu;
    this.C = kv['clip.vision.embedding_length'];
    this.F = kv['clip.vision.feed_forward_length'];
    this.nLayer = kv['clip.vision.block_count'];
    this.H = kv['clip.vision.attention.head_count'];
    this.HK = kv['clip.vision.attention.head_count_kv'];
    this.D = this.C / this.H;
    this.eps = kv['clip.vision.attention.layer_norm_epsilon'];
    this.out = kv['clip.vision.projection_dim'];
    this.cfg = {
      mean: kv['clip.vision.image_mean'], std: kv['clip.vision.image_std'],
      minPixels: kv['clip.vision.image_min_pixels'], maxPixels: kv['clip.vision.image_max_pixels'],
      patch: kv['clip.vision.patch_size'], merge: kv['clip.vision.spatial_merge_size'],
    };
    this.ropeBase = 1e6;
  }

  #u(vals, floats = []) {
    const a = new Uint32Array(vals.length), f = new Float32Array(a.buffer);
    vals.forEach((v, i) => { if (floats.includes(i)) f[i] = v; else a[i] = v; });
    return a;
  }

  #mm(enc, name, X, Y, T, bias) {
    const w = this.w[name];
    const mt = K.MAXT;
    const groups = [Math.min(w.rows, 32768), Math.ceil(w.rows / 32768), Math.ceil(T / mt)];
    const b = bias ? this.w[bias].buf : this.zero;
    if (w.type === 'Q8_0') this.gpu.dispatch(enc, src('q8', mt, K.matmulQ8_0), [this.#u([w.rows, w.K, T, w.dOff, bias ? 1 : 0]), w.buf, X, Y, b], groups);
    else this.gpu.dispatch(enc, src('f32', mt, K.matmulF32), [this.#u([w.rows, w.K, T, bias ? 1 : 0]), w.buf, X, Y, b], groups);
  }

  #ew(n) { return [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)]; }

  async #submit(enc) {
    await this.gpu.run(() => {}, enc);
  }

  /**
   * Encodes patches from image.js `patchify` into language-model embeddings. `patches2` is the second frame of a
   * video's pair (same size); a still image leaves it out.
   * @returns {Promise<{ buf: GPUBuffer, tokens: number }>} buf holds [tokens, out] floats; free it with gpu.free.
   */
  async encode({ patches, patches2, pw, ph }) {
    const gpu = this.gpu, C = this.C, F = this.F, H = this.H, HK = this.HK, D = this.D;
    const N = pw * ph, NT = N / 4;
    const f = (n) => gpu.buffer(n * 4, 'vision-act');
    const a = {
      px: f(N * 768), px2: f(N * 768), x: f(N * C), h: f(N * C), big: f(N * 3 * C), q: f(N * C), qn: f(N * C), k: f(N * HK * D), kn: f(N * HK * D),
      v: f(N * HK * D), att: f(N * C), y: f(N * C), g: f(N * F), u: f(N * F), S: f(QB * H * N),
      mrg: f(NT * 4 * C), m1: f(NT * 4 * C),
    };
    this.zero = f(4);
    const out = gpu.buffer(NT * this.out * 4, 'vision-out');
    gpu.device.queue.writeBuffer(a.px, 0, patches);
    gpu.device.queue.writeBuffer(a.px2, 0, patches2 || patches);
    try {
      // Patchify + embedding norm.
      let enc = gpu.device.createCommandEncoder();
      this.#mm(enc, 'v.patch_embd.weight', a.px, a.h, N);
      if (this.w['v.patch_embd.weight.1']) {
        this.#mm(enc, 'v.patch_embd.weight.1', a.px2, a.y, N);
        gpu.dispatch(enc, K.addInto, [this.#u([N * C]), a.y, a.h], this.#ew(N * C));
      }
      this.#norm(enc, a.h, 'v.embedding_norm.weight', a.x, N, C);
      await this.#submit(enc);
      for (let i = 0; i < this.nLayer; i++) {
        enc = gpu.device.createCommandEncoder();
        const p = `v.blk.${i}.`;
        this.#norm(enc, a.x, p + 'ln1.weight', a.h, N, C);
        if (this.w[p + 'conv2d_in_proj.weight']) {
          this.#mm(enc, p + 'conv2d_in_proj.weight', a.h, a.big, N);
          gpu.dispatch(enc, K.dwconv3, [this.#u([pw, ph, C, 0]), a.big, this.w[p + 'conv2d_spatial.weight'].buf, a.att], [Math.ceil(C / 64), N]);
          this.#mm(enc, p + 'conv2d_out_proj.weight', a.att, a.y, N);
        } else {
          this.#mm(enc, p + 'attn_q.weight', a.h, a.q, N);
          this.#mm(enc, p + 'attn_k.weight', a.h, a.k, N);
          this.#mm(enc, p + 'attn_v.weight', a.h, a.v, N);
          this.#norm(enc, a.q, p + 'attn_q_norm.weight', a.qn, N * H, D);
          this.#norm(enc, a.k, p + 'attn_k_norm.weight', a.kn, N * HK, D);
          gpu.dispatch(enc, K.rope2d, [this.#u([H, pw, this.ropeBase, 0], [2]), a.qn], N * H);
          gpu.dispatch(enc, K.rope2d, [this.#u([HK, pw, this.ropeBase, 0], [2]), a.kn], N * HK);
          for (let q0 = 0; q0 < N; q0 += QB) {
            const nq = Math.min(QB, N - q0);
            gpu.dispatch(enc, K.encScores, [this.#u([N, q0, H, HK, D, 1 / Math.sqrt(D), 0, 0], [5]), a.qn, a.kn, a.S], [Math.ceil(N / 64), nq * H]);
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
        await this.#submit(enc);
      }
      // Projector: LayerNorm → 2×2 merge → Linear → GELU → Linear.
      enc = gpu.device.createCommandEncoder();
      gpu.dispatch(enc, K.layernorm, [this.#u([C, 1e-6, 0, 0], [1]), a.x, this.w['mm.input_norm.weight'].buf, this.w['mm.input_norm.bias'].buf, a.h], N);
      gpu.dispatch(enc, K.merge2x2, [this.#u([pw, C, 0, 0]), a.h, a.mrg], NT);
      this.#mm(enc, 'mm.0.weight', a.mrg, a.m1, NT, 'mm.0.bias');
      gpu.dispatch(enc, K.geluErf, [this.#u([NT * 4 * C]), a.m1], this.#ew(NT * 4 * C));
      this.#mm(enc, 'mm.2.weight', a.m1, out, NT, 'mm.2.bias');
      await this.#submit(enc);
    } finally {
      for (const b of Object.values(a)) gpu.free(b);
      gpu.free(this.zero);
    }
    return { buf: out, tokens: NT };
  }

  #norm(enc, X, wname, Y, rows, N) {
    this.gpu.dispatch(enc, K.rmsnorm, [this.#u([N, 0, this.eps, N, 0, 0, 0, 0], [2]), X, this.w[wname].buf, Y], rows);
  }

  /** Frees the encoder's weights. */
  free() { freeTensors(this.gpu, this.w); this.w = {}; }
}
