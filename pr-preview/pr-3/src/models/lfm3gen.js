// LFM3 Omni speech output (from the mmproj): the depthformer turns the backbone's hidden state into one audio
// frame of 12 codebook codes (MaskGIT: three passes revealing 4 codebooks each), and the Liquid codec decoder turns
// frames into polar STFT frames (log magnitude + phase) that speech.js turns into 24 kHz PCM.
// Both load on demand and are freed after speaking.

import { streamTensors, freeTensors } from '../loader.js';
import * as K from '../kernels.js';

const SRC = {};
const src = (name, mt, f) => (SRC[name + mt] ||= f(mt));
const ew = (n) => [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)];

function u(vals, floats = []) {
  const a = new Uint32Array(vals.length), f = new Float32Array(a.buffer);
  vals.forEach((v, i) => { if (floats.includes(i)) f[i] = v; else a[i] = v; });
  return a;
}

// Q8_0 / F32 matmul for T tokens (+ bias).
function mm(gpu, enc, w, X, Y, T, bias, zero) {
  const mt = T === 1 ? 1 : K.MAXT;
  const groups = [Math.min(w.rows, 32768), Math.ceil(w.rows / 32768), Math.ceil(T / mt)];
  if (w.type === 'Q8_0') gpu.dispatch(enc, src('q8', mt, K.matmulQ8_0), [u([w.rows, w.K, T, w.dOff, bias ? 1 : 0]), w.buf, X, Y, bias || zero], groups);
  else gpu.dispatch(enc, src('f32', mt, K.matmulF32), [u([w.rows, w.K, T, bias ? 1 : 0]), w.buf, X, Y, bias || zero], groups);
}

export class LFM3SpeechGen {
  /** @param {import('../gpu.js').Gpu} gpu */
  static async load(gpu, source, gguf, { onProgress } = {}) {
    const tensors = gguf.tensors.filter((t) => t.name.startsWith('a.gen.'));
    if (!tensors.length) throw new Error('this mmproj cannot speak (no speech generator)');
    const g = new LFM3SpeechGen(gpu, gguf.kv);
    g.w = await streamTensors(gpu, source, tensors, { tag: 'speech', onProgress });
    g.#alloc();
    return g;
  }

  constructor(gpu, kv) {
    this.gpu = gpu;
    const k = (s) => kv[`clip.gen.audio.${s}`];
    this.E = k('embedding_length'); // 1024
    this.F = k('feed_forward_length');
    this.nLayer = k('block_count');
    this.H = k('attention.head_count');
    this.HK = k('attention.head_count_kv');
    this.D = this.E / this.H;
    this.eps = k('attention.layer_norm_epsilon');
    this.out = k('projection_dim'); // 2048
    this.sampling = { temp: k('sampling.temp') ?? 0.4, penalty: k('sampling.penalty_repeat') ?? 1.2, lastN: k('sampling.penalty_last_n') ?? 64 };
    this.nCb = 12;
    // Codec decoder (fixed by LiquidCodecConfig in llama.cpp).
    this.cE = 512; this.cEps = 1e-5; this.cRope = 1e6; this.cWindow = 30; this.stack = 6;
  }

  #alloc() {
    const gpu = this.gpu, f = (n) => gpu.buffer(n * 4, 'speech-act');
    const embd = this.w['a.gen.code.embd.weight'];
    this.nVocab = embd.rows / this.nCb; // 2017 (the last code ends the audio)
    embd.rows = this.nVocab * this.nCb; embd.K = this.E;
    const oe = this.w['a.gen.code.out_embd.weight'];
    oe.K = this.out;
    const n = this.nCb, E = this.E;
    this.a = {
      h: f(this.out), proj: f(n * E), tok: f(n * E), mask: f(n), idx: gpu.buffer(64, 'speech-act'),
      x: f(n * E), hn: f(n * E), q: f(n * E), qn: f(n * E), k: f(n * this.HK * this.D), kn: f(n * this.HK * this.D),
      v: f(n * this.HK * this.D), gate: f(n * this.H), att: f(n * E), y: f(n * E), g: f(n * this.F), u: f(n * this.F),
      S: f(n * this.H * n), logits: f(n * this.nVocab), fbRows: f(n * this.out), fb: f(this.out), zero: f(4),
    };
    // Codec: one frame = 6 positions; ring KV caches keep the 30-position window between frames.
    const P = this.stack, cE = this.cE;
    const lay = [];
    for (let i = 0; ; i++) {
      const p = `a.gen.wav.tfm.blk.${i}.`;
      if (!this.w[p + 'ln1.weight']) break;
      if (this.w[p + 'conv_dw.weight']) lay.push({ p, conv: true, state: f(6 * cE) });
      else {
        const kvW = this.w[p + 'attn_k.weight'].rows;
        lay.push({ p, conv: false, kvW, slots: this.cWindow + P, k: f((this.cWindow + P) * kvW), v: f((this.cWindow + P) * kvW) });
      }
    }
    this.cLayers = lay;
    const kvW = Math.max(...lay.filter((l) => !l.conv).map((l) => l.kvW));
    this.c = {
      codes: gpu.buffer(64, 'speech-act'), qv: f(48), q1: f(256), fr: f(cE), x: f(P * cE), h: f(P * cE), big: f(P * 3 * cE),
      att: f(P * cE), y: f(P * cE), q: f(P * cE), qn: f(P * cE), k: f(P * kvW), kn: f(P * kvW), v: f(P * kvW), gate: f(P * 16),
      S: f(P * 16 * (this.cWindow + P)), g: f(P * 2304), u: f(P * 2304), out: f(P * 1282),
    };
    this.cPos = 0;
  }

  /** Clears the codec's streaming state (start of a new utterance). */
  resetCodec() {
    this.cPos = 0;
    for (const l of this.cLayers) if (l.conv) this.gpu.device.queue.writeBuffer(l.state, 0, new Float32Array(6 * this.cE));
  }

  // mode 1 adds the normed rows into Y (sandwich post-norm + residual).
  #norm(enc, X, w, Y, rows, N, eps, mode = 0) {
    this.gpu.dispatch(enc, K.rmsnorm, [u([N, mode, eps, N, 0, 0, 0, 0], [2]), X, w, Y], rows);
  }

  /**
   * One MaskGIT pass: logits [12 × nVocab] for every codebook given the hidden state (GPU buffer, 2048 floats)
   * and the codes known so far (-1 = masked).
   */
  async codeLogits(hBuf, codes) {
    const gpu = this.gpu, a = this.a, w = (n) => this.w[n], n = this.nCb, E = this.E, H = this.H, HK = this.HK, D = this.D;
    const q = gpu.device.queue;
    q.writeBuffer(a.idx, 0, Uint32Array.from(codes, (c, i) => Math.max(0, c) + i * this.nVocab));
    q.writeBuffer(a.mask, 0, Float32Array.from(codes, (c) => (c >= 0 ? 1 : 0)));
    const enc = gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(hBuf, 0, a.h, 0, this.out * 4);
    mm(gpu, enc, w('a.gen.code.proj_in.weight'), a.h, a.proj, 1, w('a.gen.code.proj_in.bias').buf, a.zero);
    const embd = w('a.gen.code.embd.weight');
    gpu.dispatch(enc, K.gatherQ8_0, [u([E, embd.dOff, n, 0]), embd.buf, a.idx, a.tok], n);
    gpu.dispatch(enc, K.depthInput, [u([E, n, 0, 0]), a.proj, a.tok, w('a.gen.code.mask_embd').buf, a.mask, a.x], Math.ceil((n * E) / 256));
    for (let i = 0; i < this.nLayer; i++) {
      const p = `a.gen.code.blk.${i}.`;
      this.#norm(enc, a.x, w(p + 'ln1.weight').buf, a.hn, n, E, this.eps);
      mm(gpu, enc, w(p + 'attn_q.weight'), a.hn, a.q, n, null, a.zero);
      mm(gpu, enc, w(p + 'attn_k.weight'), a.hn, a.k, n, null, a.zero);
      mm(gpu, enc, w(p + 'attn_v.weight'), a.hn, a.v, n, null, a.zero);
      mm(gpu, enc, w(p + 'attn_gate.weight'), a.hn, a.gate, n, null, a.zero);
      this.#norm(enc, a.q, w(p + 'attn_q_norm.weight').buf, a.qn, n * H, D, this.eps);
      this.#norm(enc, a.k, w(p + 'attn_k_norm.weight').buf, a.kn, n * HK, D, this.eps);
      gpu.dispatch(enc, K.ropeN, [u([H, 0, 1e6, D], [2]), a.qn], n * H);
      gpu.dispatch(enc, K.ropeN, [u([HK, 0, 1e6, D], [2]), a.kn], n * HK);
      gpu.dispatch(enc, K.encScores, [u([n, 0, H, HK, D, 1 / Math.sqrt(D), 0, 0], [5]), a.qn, a.kn, a.S], [1, n * H]);
      gpu.dispatch(enc, K.softmax, [u([n, n, 0, 0]), a.S], n * H);
      gpu.dispatch(enc, K.encOut, [u([n, 0, H, HK, D, 0, 0, 0]), a.S, a.v, a.att], n * H);
      gpu.dispatch(enc, K.gateHeads, [u([H, D, n * E, 0]), a.att, a.gate], Math.ceil((n * E) / 256));
      mm(gpu, enc, w(p + 'attn_out.weight'), a.att, a.y, n, null, a.zero);
      gpu.dispatch(enc, K.addInto, [u([n * E]), a.y, a.x], ew(n * E));
      this.#norm(enc, a.x, w(p + 'ln2.weight').buf, a.hn, n, E, this.eps);
      mm(gpu, enc, w(p + 'ffn_gate.weight'), a.hn, a.g, n, null, a.zero);
      mm(gpu, enc, w(p + 'ffn_up.weight'), a.hn, a.u, n, null, a.zero);
      gpu.dispatch(enc, K.swiglu, [u([n * this.F]), a.g, a.u], ew(n * this.F));
      mm(gpu, enc, w(p + 'ffn_down.weight'), a.g, a.y, n, null, a.zero);
      gpu.dispatch(enc, K.addInto, [u([n * E]), a.y, a.x], ew(n * E));
    }
    this.#norm(enc, a.x, w('a.gen.code.output_norm.weight').buf, a.hn, n, E, this.eps);
    gpu.dispatch(enc, K.rmsnormRowW, [u([E, this.eps, 0, 0], [1]), a.hn, w('a.gen.code.head_norm.weight').buf, a.x], n);
    const groups = Math.ceil(embd.rows / 8);
    gpu.dispatch(enc, K.matvecQ8Grouped, [u([embd.rows, E, embd.dOff, this.nVocab]), embd.buf, a.x, a.logits], [Math.min(groups, 4096), Math.ceil(groups / 4096)]);
    await gpu.run(() => {}, enc);
    return new Float32Array(await gpu.read(a.logits, n * this.nVocab * 4));
  }

  /** Feedback embedding of a full frame (sum of the codes' output embeddings), as a GPU buffer of 2048 floats. */
  async feedback(codes) {
    const gpu = this.gpu, a = this.a, oe = this.w['a.gen.code.out_embd.weight'], n = this.nCb;
    gpu.device.queue.writeBuffer(a.idx, 0, Uint32Array.from(codes, (c, i) => c + i * this.nVocab));
    await gpu.run((enc) => {
      gpu.dispatch(enc, K.gatherQ8_0, [u([this.out, oe.dOff, n, 0]), oe.buf, a.idx, a.fbRows], n);
      gpu.dispatch(enc, K.sumRows, [u([this.out, n, 0, 0]), a.fbRows, a.fb], Math.ceil(this.out / 256));
    });
    return a.fb;
  }

  /** Codec: one frame of 12 codes → 6 STFT frames [6 × 1282] (641 log magnitudes, then 641 phases each). */
  async decodeFrame(codes) {
    const gpu = this.gpu, c = this.c, w = (n) => this.w[n], P = this.stack, E = this.cE, pos0 = this.cPos;
    gpu.device.queue.writeBuffer(c.codes, 0, Uint32Array.from(codes));
    const enc = gpu.device.createCommandEncoder();
    const z = this.a.zero;
    gpu.dispatch(enc, K.gatherF32, [u([4, this.nCb, 0, 0]), w('a.gen.wav.quant.first.codebook.weight').buf, c.codes, c.qv], 1);
    mm(gpu, enc, w('a.gen.wav.quant.first.out_proj.weight'), c.qv, c.q1, 1, w('a.gen.wav.quant.first.out_proj.bias').buf, z);
    mm(gpu, enc, w('a.gen.wav.tfm.in_proj.weight'), c.q1, c.fr, 1, w('a.gen.wav.tfm.in_proj.bias').buf, z);
    gpu.dispatch(enc, K.repeatRows, [u([E, P, P * E, 0]), c.fr, c.x], Math.ceil((P * E) / 256));
    for (const L of this.cLayers) {
      const p = L.p;
      this.#norm(enc, c.x, w(p + 'ln1.weight').buf, c.h, P, E, this.cEps);
      if (L.conv) {
        mm(gpu, enc, w(p + 'conv_pw1.weight'), c.h, c.big, P, null, z);
        gpu.dispatch(enc, K.shortconv, [u([P, E, 0, 0]), c.big, w(p + 'conv_dw.weight').buf, L.state, c.att], Math.ceil(E / 64));
        mm(gpu, enc, w(p + 'conv_pw2.weight'), c.att, c.y, P, null, z);
      } else {
        const D = w(p + 'attn_q_norm.weight').K, H = E / D, HK = L.kvW / D;
        mm(gpu, enc, w(p + 'attn_q.weight'), c.h, c.q, P, null, z);
        mm(gpu, enc, w(p + 'attn_k.weight'), c.h, c.k, P, null, z);
        mm(gpu, enc, w(p + 'attn_v.weight'), c.h, c.v, P, null, z);
        mm(gpu, enc, w(p + 'attn_gate.weight'), c.h, c.gate, P, null, z);
        this.#norm(enc, c.q, w(p + 'attn_q_norm.weight').buf, c.qn, P * H, D, this.cEps);
        this.#norm(enc, c.k, w(p + 'attn_k_norm.weight').buf, c.kn, P * HK, D, this.cEps);
        gpu.dispatch(enc, K.ropeN, [u([H, pos0, this.cRope, D], [2]), c.qn], P * H);
        gpu.dispatch(enc, K.ropeN, [u([HK, pos0, this.cRope, D], [2]), c.kn], P * HK);
        gpu.dispatch(enc, K.ringWrite, [u([pos0, L.slots, L.kvW, 0]), c.kn, L.k], P);
        gpu.dispatch(enc, K.ringWrite, [u([pos0, L.slots, L.kvW, 0]), c.v, L.v], P);
        const Lk = Math.min(pos0 + P, L.slots);
        gpu.dispatch(enc, K.ringScores, [u([pos0, P, L.slots, Lk, this.cWindow, H, HK, D]), c.qn, L.k, c.S], [Math.ceil(Lk / 64), P * H]);
        gpu.dispatch(enc, K.softmax, [u([Lk, Lk, 0, 0]), c.S], P * H);
        gpu.dispatch(enc, K.ringOut, [u([Lk, H, HK, D]), c.S, L.v, c.att], P * H);
        gpu.dispatch(enc, K.gateHeads, [u([H, D, P * E, 0]), c.att, c.gate], Math.ceil((P * E) / 256));
        mm(gpu, enc, w(p + 'attn_out.weight'), c.att, c.y, P, null, z);
      }
      this.#norm(enc, c.y, w(p + 'attn_post_norm.weight').buf, c.x, P, E, this.cEps, 1);
      this.#norm(enc, c.x, w(p + 'ln2.weight').buf, c.h, P, E, this.cEps);
      mm(gpu, enc, w(p + 'ffn_gate.weight'), c.h, c.g, P, null, z);
      mm(gpu, enc, w(p + 'ffn_up.weight'), c.h, c.u, P, null, z);
      gpu.dispatch(enc, K.swiglu, [u([P * 2304]), c.g, c.u], ew(P * 2304));
      mm(gpu, enc, w(p + 'ffn_down.weight'), c.g, c.y, P, null, z);
      this.#norm(enc, c.y, w(p + 'ffn_post_norm.weight').buf, c.x, P, E, this.cEps, 1);
    }
    this.#norm(enc, c.x, w('a.gen.wav.tfm.output_norm.weight').buf, c.h, P, E, this.cEps);
    mm(gpu, enc, w('a.gen.wav.tfm.out_proj.weight'), c.h, c.out, P, w('a.gen.wav.tfm.out_proj.bias').buf, z);
    await gpu.run(() => {}, enc);
    this.cPos += P;
    return new Float32Array(await gpu.read(c.out, P * 1282 * 4));
  }

  /** Frees the generator's weights and buffers. */
  free() {
    freeTensors(this.gpu, this.w); this.w = {};
    for (const b of [...Object.values(this.a), ...Object.values(this.c)]) this.gpu.free(b);
    for (const l of this.cLayers) for (const b of [l.state, l.k, l.v]) if (b) this.gpu.free(b);
  }
}
