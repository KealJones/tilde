// Qwen3 (dense): pre-norm GQA attention with per-head RMSNorm on q and k, NeoX RoPE, SwiGLU FFN, Q4_0 or Q6_K
// embedding/head (or a separate head). Same kernels as LFM3 (src/models/lfm3.js), with the attention shapes as parameters and no
// output gate. Matches llama.cpp's qwen3.cpp.

import { readGGUF, archKey } from '../gguf.js';
import { streamTensors } from '../loader.js';
import * as K from '../kernels.js';

// Prompt chunks of this many tokens or more use the fused attention kernel (K.attnFlashFor); smaller ones and decode use
// scores → softmax → out, which spreads one token's work over the keys. The score buffer only needs room for those.
const FLASH_FROM = 33;

const MM = {}; // matmul source by tokens-per-pass

export class Qwen3 {
  static seqs = 0;
  /**
   * Streams a Qwen3 GGUF into GPU memory. Layer weights must be Q4_0; the embedding and head Q4_0 or Q6_K.
   * @param {import('../gpu.js').Gpu} gpu
   * @param {import('../gguf.js').ByteSource} src
   * @param {{ ctx?: number, chunk?: number, maxChunk?: number, submitBudgetMs?: number, gguf?: object, onProgress?: (p: {loaded: number, total: number, tensor: string}) => void }} opts
   */
  static async load(gpu, src, { ctx = 4096, chunk = 64, maxChunk = chunk, submitBudgetMs = 1000, gguf, onProgress } = {}) {
    const g = gguf || await readGGUF(src);
    if (g.kv['general.architecture'] !== 'qwen3') throw new Error(`not a Qwen3 model: ${g.kv['general.architecture']}`);
    for (const t of g.tensors) {
      const want = t.name === 'token_embd.weight' || t.name === 'output.weight' ? ['Q4_0', 'Q6_K'] : t.dims.length > 1 ? ['Q4_0'] : ['F32'];
      if (!want.includes(t.typeName)) throw new Error(`Tilde runs Qwen3 with ${want.join(' or ')} for ${t.name}; this file has ${t.typeName} (a pure Q4_0 GGUF works, e.g. ggml-org/Qwen3-0.6B-GGUF)`);
    }
    const m = new Qwen3(gpu, g, { ctx, chunk, maxChunk, submitBudgetMs });
    m.w = await streamTensors(gpu, src, g.tensors, { tag: 'weights', onProgress });
    m.#allocState();
    return m;
  }

  constructor(gpu, g, { ctx, chunk, maxChunk = chunk, submitBudgetMs = 1000 }) {
    this.gpu = gpu;
    this.gguf = g;
    this.ctx = ctx;
    /** Tokens per forward pass now; adapts between 8 and maxChunk (activations are sized for maxChunk). */
    this.chunk = chunk;
    this.maxChunk = Math.max(chunk, maxChunk);
    this.nLayer = archKey(g, 'block_count');
    this.nEmbd = archKey(g, 'embedding_length');
    this.nFF = archKey(g, 'feed_forward_length');
    this.eps = archKey(g, 'attention.layer_norm_rms_epsilon');
    this.ropeBase = archKey(g, 'rope.freq_base');
    this.nHead = archKey(g, 'attention.head_count');
    this.nKV = archKey(g, 'attention.head_count_kv');
    const hd = archKey(g, 'attention.key_length') ?? this.nEmbd / this.nHead;
    if (hd !== 128) throw new Error(`Tilde's attention kernels need 128-wide heads; this model has ${hd}`);
    if (this.nHead % this.nKV) throw new Error('query heads must be a multiple of KV heads');
    this.qDim = this.nHead * 128;
    this.kvDim = this.nKV * 128;
    this.K = {
      kvWrite: K.kvWriteFor(this.nKV),
      scores: K.attnScoresFor(this.nHead, this.nKV),
      out: K.attnOutFor(this.nHead, this.nKV, false),
      flash: K.attnFlashFor(this.nHead, this.nKV, false),
    };
    this.tokenizer = {
      tokens: g.kv['tokenizer.ggml.tokens'], merges: g.kv['tokenizer.ggml.merges'],
      bos: g.kv['tokenizer.ggml.bos_token_id'], eos: g.kv['tokenizer.ggml.eos_token_id'],
      template: g.kv['tokenizer.chat_template'],
    };
    this.vocab = this.tokenizer.tokens.length;
    this.w = {};
    this.pos = 0;
    /** Submits are sized to stay under this many ms (see LFM3: WebKit kills the GPU at ~3 s). */
    this.submitBudgetMs = submitBudgetMs;
    /** Single-token matmul: 'wide' (8 rows per workgroup) or 'row' (one workgroup per row). */
    this.decodeKernel = 'wide';
    /** GPU ms per layer per token, by batch size rounded up to a power of 2. */
    this.est = new Map();
  }

  #allocState() {
    const gpu = this.gpu, T = this.maxChunk, E = this.nEmbd, F = this.nFF;
    const f = (n, tag = 'activations') => gpu.buffer(n * 4, tag);
    this.a = {
      x: f(T * E), h: f(T * E), y: f(T * E),
      q: f(T * this.qDim), qn: f(T * this.qDim), att: f(T * this.qDim),
      k: f(T * this.kvDim), kn: f(T * this.kvDim), v: f(T * this.kvDim),
      g: f(T * F), u: f(T * F), S: f(Math.min(T, FLASH_FROM - 1) * this.nHead * this.ctx), headIn: f(E), logits: f(this.vocab),
      tok: gpu.buffer(T * 4, 'activations'), ones: f(64),
    };
    gpu.device.queue.writeBuffer(this.a.ones, 0, new Float32Array(64).fill(1));
    this.cache = this.#allocCache(this.ctx);
    this.seq = 0;
  }

  // KV for every layer: ctx slots each (no sliding window, no conv state).
  #allocCache(ctx) {
    return Array.from({ length: this.nLayer }, () => ({
      slots: ctx, k: this.gpu.buffer(ctx * this.kvDim * 2, 'kv'), v: this.gpu.buffer(ctx * this.kvDim * 2, 'kv'), // f16
    }));
  }

  /** A second conversation state (its own KV, `ctx` positions) to switch to with useSequence. */
  newSequence(ctx) {
    if (ctx > this.ctx) throw new Error('a side sequence can be at most as long as the model\'s context');
    return { id: ++Qwen3.seqs, cache: this.#allocCache(ctx), ctx, pos: 0 };
  }

  /** Switches to sequence `s` and returns the one that was in use. */
  useSequence(s) {
    const was = { id: this.seq, cache: this.cache, ctx: this.ctx, pos: this.pos };
    Object.assign(this, { seq: s.id, cache: s.cache, ctx: s.ctx, pos: s.pos });
    return was;
  }

  /** Frees a sequence from newSequence (not the one in use). */
  freeSequence(s) {
    for (const c of s.cache) for (const b of [c.k, c.v]) this.gpu.free(b);
  }

  /** Remembers the current position. KV slots before it are never written again, so that is the whole state. */
  async snapshot() { this.snap = { seq: this.seq, pos: this.pos }; }

  /** Goes back to the last snapshot() of this sequence. */
  async restore() {
    if (!this.snap || this.snap.seq !== this.seq) throw new Error('no snapshot of this sequence');
    this.pos = this.snap.pos;
  }

  /** Forgets the conversation. */
  reset() { this.pos = 0; }

  #u(...vals) { return Uint32Array.from(vals); }
  #uf(vals, floats) {
    const a = new Uint32Array(vals.length), f = new Float32Array(a.buffer);
    vals.forEach((v, i) => { if (floats.includes(i)) f[i] = v; else a[i] = v; });
    return a;
  }

  #matmul(enc, name, X, Y, T) {
    const w = this.w[name];
    const rows = w.rows;
    const u = this.#u(rows, w.K, T, w.dOff, rows, 0, w.K, 0);
    if (T >= 4) {
      // 64×64 tiles from 33 tokens up (3× gemmQ4_0 on an iPhone 16, bench/kernel-bench.html); 32×64 below.
      if (T > 32) this.gpu.dispatch(enc, K.gemmQ4_0x64, [u, w.buf, X, Y], [Math.ceil(rows / 64), Math.ceil(T / 64)]);
      else this.gpu.dispatch(enc, K.gemmQ4_0, [u, w.buf, X, Y], [Math.ceil(rows / 64), Math.ceil(T / 32)]);
      return;
    }
    if (T === 1 && this.decodeKernel === 'wide') {
      const g = Math.ceil(rows / 8);
      this.gpu.dispatch(enc, K.matvecQ4_0, [u, w.buf, X, Y], [Math.min(g, 4096), Math.ceil(g / 4096)]);
      return;
    }
    const mt = T === 1 ? 1 : K.MAXT;
    this.gpu.dispatch(enc, (MM[mt] ||= K.matmulQ4_0(mt)), [u, w.buf, X, Y], [Math.min(rows, 32768), Math.ceil(rows / 32768), Math.ceil(T / mt)]);
  }
  #norm(enc, X, w, Y, rows, N, { xOff = 0 } = {}) {
    this.gpu.dispatch(enc, K.rmsnorm, [this.#uf([N, 0, this.eps, N, xOff, 0, 0, 0], [2]), X, w, Y], rows);
  }
  #add(enc, X, Y, n) {
    this.gpu.dispatch(enc, K.addInto, [this.#u(n, 0, 0, 0), X, Y], [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)]);
  }

  /** Runs tokens (≤ chunk) through the model at the current position, in submits sized by measurement. */
  async forward(tokens) {
    const T = tokens.length, E = this.nEmbd, a = this.a, gpu = this.gpu, pos0 = this.pos;
    if (T > this.maxChunk) throw new Error(`forward takes at most ${this.maxChunk} tokens`);
    if (pos0 + T > this.ctx) throw new Error('context is full');
    const bucket = 2 ** Math.ceil(Math.log2(T));
    let est = this.est.get(bucket);
    if (est === undefined) est = Math.max(20, ...[...this.est].filter(([b]) => b < bucket).map(([, v]) => v));
    const per = Math.max(1, Math.min(this.nLayer, Math.floor(this.submitBudgetMs / (est * T))));
    gpu.beginPlan(T === 1 ? `qwen3-decode:${this.seq}` : null);
    let enc = gpu.device.createCommandEncoder();
    gpu.device.queue.writeBuffer(a.tok, 0, new Uint32Array(tokens));
    const emb = this.w['token_embd.weight'];
    if (emb.type === 'Q6_K') gpu.dispatch(enc, K.embedQ6_K, [this.#u(E, T, 0, 0), emb.buf, a.tok, a.x], T);
    else gpu.dispatch(enc, K.embedQ4_0, [this.#u(E, emb.dOff, 0, 0), emb.buf, a.tok, a.x], T);
    let inEnc = 0;
    const learn = (ms, layers) => {
      const e = ms / (layers * T);
      est = e > est ? e : 0.8 * est + 0.2 * e;
      this.est.set(bucket, est);
    };
    for (let i = 0; i < this.nLayer; i++) {
      this.#layer(enc, i, T, pos0);
      inEnc++;
      if (inEnc === per && i + 1 < this.nLayer) {
        const ms = await this.gpu.run(() => {}, enc);
        if (T > 1) learn(ms, inEnc);
        inEnc = 0;
        enc = gpu.device.createCommandEncoder();
      }
    }
    this.#norm(enc, a.x, this.w['output_norm.weight'].buf, a.headIn, 1, E, { xOff: (T - 1) * E });
    const headName = this.w['output.weight'] ? 'output.weight' : 'token_embd.weight';
    const head = this.w[headName];
    if (head.type === 'Q6_K') gpu.dispatch(enc, K.matmulQ6_K, [this.#u(head.rows, E, 0, 0), head.buf, a.headIn, a.logits], [Math.min(head.rows, 32768), Math.ceil(head.rows / 32768)]);
    else this.#matmul(enc, headName, a.headIn, a.logits, 1);
    const ms = await this.gpu.run(() => {}, enc);
    gpu.beginPlan(null);
    if (T > 1 && inEnc) learn(ms, inEnc);
    const big = this.est.get(2 ** Math.ceil(Math.log2(this.chunk)));
    if (big !== undefined && big * this.chunk > this.submitBudgetMs && this.chunk > 8) this.chunk >>= 1;
    else if (big !== undefined && big * this.chunk * 2 < this.submitBudgetMs / 2 && this.chunk * 2 <= this.maxChunk) this.chunk *= 2;
    this.pos += T;
  }

  /** Qwen3 has no media input. */
  async forwardEmbeds() { throw new Error('Qwen3 takes text only'); }

  #layer(enc, i, T, pos0) {
    const E = this.nEmbd, F = this.nFF, H = this.nHead, KV = this.nKV, a = this.a, gpu = this.gpu;
    const p = `blk.${i}.`, c = this.cache[i];
    this.#norm(enc, a.x, this.w[p + 'attn_norm.weight'].buf, a.h, T, E);
    this.#matmul(enc, p + 'attn_q.weight', a.h, a.q, T);
    this.#matmul(enc, p + 'attn_k.weight', a.h, a.k, T);
    this.#matmul(enc, p + 'attn_v.weight', a.h, a.v, T);
    this.#norm(enc, a.q, this.w[p + 'attn_q_norm.weight'].buf, a.qn, T * H, 128);
    this.#norm(enc, a.k, this.w[p + 'attn_k_norm.weight'].buf, a.kn, T * KV, 128);
    gpu.dispatch(enc, K.rope, [this.#uf([H, pos0, this.ropeBase, 0], [2]), a.qn, a.ones], T * H);
    gpu.dispatch(enc, K.rope, [this.#uf([KV, pos0, this.ropeBase, 0], [2]), a.kn, a.ones], T * KV);
    gpu.dispatch(enc, this.K.kvWrite, [this.#u(pos0, c.slots, 0, 0), a.kn, a.v, c.k, c.v], T);
    const Lk = pos0 + T;
    if (T >= FLASH_FROM) {
      gpu.dispatch(enc, this.K.flash, [this.#u(pos0, T, c.slots, Lk, 0xffffffff, 0, 0, 0), a.qn, c.k, c.v, a.att], [Math.ceil(T / (16 / (H / KV))), KV]);
    } else {
      gpu.dispatch(enc, this.K.scores, [this.#u(pos0, T, c.slots, Lk, 0xffffffff, this.ctx, 0, 0), a.qn, c.k, a.S], [Math.ceil(Lk / 64), T * H]);
      gpu.dispatch(enc, K.softmax, [this.#u(Lk, this.ctx, 0, 0), a.S], T * H);
      gpu.dispatch(enc, this.K.out, [this.#u(Lk, this.ctx, 0, 0), a.S, c.v, a.att], T * H);
    }
    this.#matmul(enc, p + 'attn_output.weight', a.att, a.y, T);
    this.#add(enc, a.y, a.x, T * E);
    this.#norm(enc, a.x, this.w[p + 'ffn_norm.weight'].buf, a.h, T, E);
    this.#matmul(enc, p + 'ffn_gate.weight', a.h, a.g, T);
    this.#matmul(enc, p + 'ffn_up.weight', a.h, a.u, T);
    const n = T * F;
    gpu.dispatch(enc, K.swiglu, [this.#u(n, 0, 0, 0), a.g, a.u], [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)]);
    this.#matmul(enc, p + 'ffn_down.weight', a.g, a.y, T);
    this.#add(enc, a.y, a.x, T * E);
  }

  /** Top candidates of the last token's logits, computed on the GPU: { values, ids }. */
  async candidates() {
    const gpu = this.gpu, n = Math.ceil(this.vocab / 256);
    this.cand ||= gpu.buffer(n * 16 * 8, 'activations');
    gpu.beginPlan('qwen3-topk');
    await gpu.run((enc) => gpu.dispatch(enc, K.topk16, [this.#u(this.vocab, 0, 0, 0), this.a.logits, this.cand], n));
    gpu.beginPlan(null);
    const u = new Uint32Array(await gpu.read(this.cand, n * 16 * 8));
    const values = new Float32Array(n * 16), ids = new Uint32Array(n * 16);
    const f = new Float32Array(u.buffer);
    for (let i = 0; i < n * 16; i++) { values[i] = f[i * 2]; ids[i] = u[i * 2 + 1]; }
    return { values, ids };
  }

  /** Logits of the last token from the latest forward(). */
  async logits() {
    return new Float32Array(await this.gpu.read(this.a.logits, this.vocab * 4));
  }
}
