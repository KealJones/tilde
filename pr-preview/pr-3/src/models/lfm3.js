// LFM3 (Liquid AI): 24 layers of short conv / gated GQA attention (some sliding window), sandwich norms,
// SwiGLU FFN, tied Q6_K embedding/head. See docs/design.md for the math.

import { readGGUF, archKey } from '../gguf.js';
import { streamTensors } from '../loader.js';
import * as K from '../kernels.js';

// Prompt chunks of this many tokens or more use the fused attention kernel (K.attnFlashFor); smaller ones and decode use
// scores → softmax → out, which spreads one token's work over the keys. The score buffer only needs room for those.
const FLASH_FROM = 33;

const MM = {}; // matmul source by tokens-per-pass

export class LFM3 {
  static seqs = 0;
  /**
   * Streams an LFM3 GGUF into GPU memory.
   * @param {import('../gpu.js').Gpu} gpu
   * @param {import('../gguf.js').ByteSource} src
   * @param {{ ctx?: number, chunk?: number, maxChunk?: number, submitBudgetMs?: number, gguf?: object, onProgress?: (p: {loaded: number, total: number, tensor: string}) => void }} opts
   */
  static async load(gpu, src, { ctx = 4096, chunk = 64, maxChunk = chunk, submitBudgetMs = 1000, gguf, onProgress } = {}) {
    const g = gguf || await readGGUF(src);
    if (g.kv['general.architecture'] !== 'lfm3') throw new Error(`not an LFM3 model: ${g.kv['general.architecture']}`);
    const m = new LFM3(gpu, g, { ctx, chunk, maxChunk, submitBudgetMs });
    await m.#stream(src, onProgress);
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
    this.ropeBaseSWA = archKey(g, 'rope.freq_base_swa');
    this.window = archKey(g, 'attention.sliding_window');
    const kvHeads = archKey(g, 'attention.head_count_kv');
    const swa = archKey(g, 'attention.sliding_window_pattern');
    this.layers = Array.from({ length: this.nLayer }, (_, i) => ({ attn: kvHeads[i] > 0, swa: !!swa[i] }));
    this.tokenizer = {
      tokens: g.kv['tokenizer.ggml.tokens'], merges: g.kv['tokenizer.ggml.merges'],
      bos: g.kv['tokenizer.ggml.bos_token_id'], eos: g.kv['tokenizer.ggml.eos_token_id'],
      template: g.kv['tokenizer.chat_template'],
    };
    this.vocab = this.tokenizer.tokens.length;
    this.w = {};
    this.pos = 0;
    /**
     * Each submit is sized ahead of time to stay under `submitBudgetMs`, from a running estimate of GPU ms per
     * layer per token. The estimate starts pessimistic and follows the slowest recent submit, so a cold device
     * starts with small submits instead of finding out the hard way (WebKit kills the GPU at ~3 s).
     */
    this.submitBudgetMs = submitBudgetMs;
    /** Single-token matmul: 'wide' (8 rows per workgroup) or 'row' (one workgroup per row). */
    this.decodeKernel = 'wide';
    /** Estimates by batch size (rounded up to a power of 2): cost per token is not flat in the batch size. */
    this.est = new Map();
  }

  async #stream(src, onProgress) {
    this.w = await streamTensors(this.gpu, src, this.gguf.tensors, { tag: 'weights', onProgress });
  }

  #allocState() {
    const gpu = this.gpu, T = this.maxChunk, E = this.nEmbd, F = this.nFF;
    const f = (n, tag = 'activations') => gpu.buffer(n * 4, tag);
    this.swaSlots = this.window + T;
    this.a = {
      x: f(T * E), h: f(T * E), y: f(T * E), att: f(T * E), bcx: f(T * 3 * E),
      q: f(T * E), qn: f(T * E), k: f(T * 512), kn: f(T * 512), v: f(T * 512), gate: f(T * 16),
      g: f(T * F), u: f(T * F), S: f(Math.min(T, FLASH_FROM - 1) * 16 * this.ctx), headIn: f(E), logits: f(this.vocab),
      tok: gpu.buffer(T * 4, 'activations'), ones: f(64),
    };
    gpu.device.queue.writeBuffer(this.a.ones, 0, new Float32Array(64).fill(1));
    this.cache = this.#allocCache(this.ctx);
    this.seq = 0;
  }

  // Per-layer state of one sequence: KV (global layers: ctx slots; sliding-window layers: a ring) or conv state.
  #allocCache(ctx) {
    const f = (n, tag) => this.gpu.buffer(n * 4, tag);
    return this.layers.map((L) => {
      if (L.attn) {
        const slots = L.swa ? this.swaSlots : ctx;
        return { slots, swa: L.swa, k: f(slots * 256, 'kv'), v: f(slots * 256, 'kv') }; // f16 pairs
      }
      return { state: f(6 * this.nEmbd, 'conv-state') };
    });
  }

  /**
   * A second conversation state (its own KV and conv state, `ctx` positions) to switch to with useSequence, e.g.
   * for speech or a transcription, so the main conversation's cache survives it. Activations are shared.
   */
  newSequence(ctx) {
    if (ctx > this.ctx) throw new Error('a side sequence can be at most as long as the model\'s context');
    const s = { id: ++LFM3.seqs, cache: this.#allocCache(ctx), ctx, pos: 0 };
    this.#zeroConv(s.cache);
    return s;
  }

  /** Switches to sequence `s` (from newSequence, or one returned here) and returns the one that was in use. */
  useSequence(s) {
    const was = { id: this.seq, cache: this.cache, ctx: this.ctx, pos: this.pos };
    Object.assign(this, { seq: s.id, cache: s.cache, ctx: s.ctx, pos: s.pos });
    return was;
  }

  /** Frees a sequence from newSequence (not the one in use). */
  freeSequence(s) {
    for (const c of s.cache) for (const b of [c.k, c.v, c.state]) if (b) this.gpu.free(b);
  }

  /**
   * Remembers the state at the current position, to come back to with restore(): the conv states and the
   * sliding-window rings (later tokens overwrite those); global KV slots up to here are never written again.
   */
  async snapshot() {
    const gpu = this.gpu;
    if (!this.snap || this.snap.seq !== this.seq) {
      if (this.snap) for (const b of this.snap.bufs) if (b) gpu.free(b);
      const bufs = this.cache.map((c) => (c.state ? gpu.buffer(c.state.size, 'kv-snapshot')
        : c.swa ? [gpu.buffer(c.k.size, 'kv-snapshot'), gpu.buffer(c.v.size, 'kv-snapshot')] : [])).flat();
      this.snap = { seq: this.seq, bufs, pos: 0 };
    }
    this.#copyState(true);
    this.snap.pos = this.pos;
    await gpu.done();
  }

  /** Goes back to the last snapshot() of this sequence. */
  async restore() {
    if (!this.snap || this.snap.seq !== this.seq) throw new Error('no snapshot of this sequence');
    this.#copyState(false);
    this.pos = this.snap.pos;
    await this.gpu.done();
  }

  #copyState(save) {
    const enc = this.gpu.device.createCommandEncoder();
    let j = 0;
    for (const c of this.cache) {
      for (const b of c.state ? [c.state] : c.swa ? [c.k, c.v] : []) {
        const s = this.snap.bufs[j++];
        if (save) enc.copyBufferToBuffer(b, 0, s, 0, b.size); else enc.copyBufferToBuffer(s, 0, b, 0, b.size);
      }
    }
    this.gpu.device.queue.submit([enc.finish()]);
  }

  #zeroConv(cache) {
    const q = this.gpu.device.queue;
    for (const c of cache) if (c.state) q.writeBuffer(c.state, 0, new Float32Array(6 * this.nEmbd));
  }

  /** Forgets the conversation (positions, KV, conv state). */
  reset() {
    this.pos = 0;
    this.#zeroConv(this.cache);
  }

  // Uniform values for one dispatch (they go into the submit's uniform slots, see Gpu.dispatch).
  #u(...vals) { return Uint32Array.from(vals); }
  #uf(vals, floats) {
    const a = new Uint32Array(vals.length), f = new Float32Array(a.buffer);
    vals.forEach((v, i) => { if (floats.includes(i)) f[i] = v; else a[i] = v; });
    return a;
  }

  #matmul(enc, name, X, Y, T, { yStride, yOff = 0, xStride, accumulate = 0 } = {}) {
    const w = this.w[name];
    const rows = w.rows;
    const u = this.#u(rows, w.K, T, w.dOff, yStride ?? rows, yOff, xStride ?? w.K, accumulate);
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
  #norm(enc, X, w, Y, rows, N, { mode = 0, xOff = 0, yOff = 0 } = {}) {
    this.gpu.dispatch(enc, K.rmsnorm, [this.#uf([N, mode, this.eps, N, xOff, yOff, 0, 0], [2]), X, w, Y], rows);
  }

  /**
   * Runs tokens (≤ chunk) through the model at the current position.
   * Layers are split across several GPU submits; if one takes longer than `submitBudgetMs`, later forwards
   * use fewer layers per submit, so no command buffer gets near WebKit's ~3 s watchdog.
   */
  async forward(tokens) { return this.#forward(tokens.length, tokens, null); }

  /**
   * Runs precomputed input embeddings (e.g. image or audio tokens from an encoder) through the model:
   * rows row0 … row0+T-1 of `buf`, each nEmbd floats.
   */
  async forwardEmbeds(buf, row0, T) { return this.#forward(T, null, { buf, row0 }); }

  async #forward(T, tokens, embeds) {
    const E = this.nEmbd, a = this.a, gpu = this.gpu, pos0 = this.pos;
    if (T > this.maxChunk) throw new Error(`forward takes at most ${this.maxChunk} tokens`);
    if (pos0 + T > this.ctx) throw new Error('context is full');
    const bucket = 2 ** Math.ceil(Math.log2(T));
    // Unknown batch size: borrow the worst estimate from a smaller one, else start pessimistic.
    let est = this.est.get(bucket);
    if (est === undefined) est = Math.max(20, ...[...this.est].filter(([b]) => b < bucket).map(([, v]) => v));
    const per = Math.max(1, Math.min(this.nLayer, Math.floor(this.submitBudgetMs / (est * T))));
    // Decode steps repeat the same dispatches on the same buffers, so their bind groups are reused.
    gpu.beginPlan(T === 1 ? `lfm3-decode:${this.seq}` : null);
    let enc = gpu.device.createCommandEncoder();
    if (tokens) {
      gpu.device.queue.writeBuffer(a.tok, 0, new Uint32Array(tokens));
      gpu.dispatch(enc, K.embedQ6_K, [this.#u(E, T, 0, 0), this.w['token_embd.weight'].buf, a.tok, a.x], T);
    } else {
      enc.copyBufferToBuffer(embeds.buf, embeds.row0 * E * 4, a.x, 0, T * E * 4);
    }
    let inEnc = 0;
    const learn = (ms, layers) => {
      // Follow slowdowns at once, speedups slowly (thermal throttling only goes one way at a time).
      const e = ms / (layers * T);
      est = e > est ? e : 0.8 * est + 0.2 * e;
      this.est.set(bucket, est);
    };
    for (let i = 0; i < this.nLayer; i++) {
      this.#layer(enc, i, T, pos0);
      inEnc++;
      if (inEnc === per && i + 1 < this.nLayer) {
        const ms = await this.#submit(enc);
        if (T > 1) learn(ms, inEnc);
        inEnc = 0;
        enc = gpu.device.createCommandEncoder();
      }
    }
    this.#norm(enc, a.x, this.w['output_norm.weight'].buf, a.headIn, 1, E, { xOff: (T - 1) * E });
    const head = this.w['token_embd.weight'];
    gpu.dispatch(enc, K.matmulQ6_K, [this.#u(head.rows, E, 0, 0), head.buf, a.headIn, a.logits], [Math.min(head.rows, 32768), Math.ceil(head.rows / 32768)]);
    const ms = await this.#submit(enc);
    gpu.beginPlan(null);
    if (T > 1 && inEnc) learn(ms, inEnc);
    // Even one layer per submit would be over budget: read fewer tokens at a time (and more again when it's fast).
    const big = this.est.get(2 ** Math.ceil(Math.log2(this.chunk)));
    if (big !== undefined && big * this.chunk > this.submitBudgetMs && this.chunk > 8) this.chunk >>= 1;
    else if (big !== undefined && big * this.chunk * 2 < this.submitBudgetMs / 2 && this.chunk * 2 <= this.maxChunk) this.chunk *= 2;
    this.pos += T;
  }

  async #submit(enc) {
    return this.gpu.run(() => {}, enc);
  }

  #layer(enc, i, T, pos0) {
    const E = this.nEmbd, F = this.nFF, a = this.a, gpu = this.gpu;
    const L = this.layers[i], p = `blk.${i}.`, c = this.cache[i];
    this.#norm(enc, a.x, this.w[p + 'attn_norm.weight'].buf, a.h, T, E);
    if (!L.attn) {
      this.#matmul(enc, p + 'shortconv.in_proj.weight', a.h, a.bcx, T);
      gpu.dispatch(enc, K.shortconv, [this.#u(T, E, 0, 0), a.bcx, this.w[p + 'shortconv.conv.weight'].buf, c.state, a.att], Math.ceil(E / 64));
      this.#matmul(enc, p + 'shortconv.out_proj.weight', a.att, a.y, T);
    } else {
      this.#matmul(enc, p + 'attn_q.weight', a.h, a.q, T);
      this.#matmul(enc, p + 'attn_k.weight', a.h, a.k, T);
      this.#matmul(enc, p + 'attn_v.weight', a.h, a.v, T);
      this.#matmul(enc, p + 'attn_gate.weight', a.h, a.gate, T);
      this.#norm(enc, a.q, this.w[p + 'attn_q_norm.weight'].buf, a.qn, T * 16, 128);
      this.#norm(enc, a.k, this.w[p + 'attn_k_norm.weight'].buf, a.kn, T * 4, 128);
      const base = L.swa ? this.ropeBaseSWA : this.ropeBase;
      const factors = L.swa ? a.ones : this.w['rope_freqs.weight'].buf;
      gpu.dispatch(enc, K.rope, [this.#uf([16, pos0, base, 0], [2]), a.qn, factors], T * 16);
      gpu.dispatch(enc, K.rope, [this.#uf([4, pos0, base, 0], [2]), a.kn, factors], T * 4);
      gpu.dispatch(enc, K.kvWrite, [this.#u(pos0, c.slots, 0, 0), a.kn, a.v, c.k, c.v], T);
      const Lk = Math.min(pos0 + T, c.slots);
      const win = L.swa ? this.window : 0xffffffff;
      if (T >= FLASH_FROM) {
        gpu.dispatch(enc, K.attnFlash, [this.#u(pos0, T, c.slots, Lk, win, 0, 0, 0), a.qn, c.k, c.v, a.gate, a.att], [Math.ceil(T / 4), 4]);
      } else {
        gpu.dispatch(enc, K.attnScores, [this.#u(pos0, T, c.slots, Lk, win, this.ctx, 0, 0), a.qn, c.k, a.S], [Math.ceil(Lk / 64), T * 16]);
        gpu.dispatch(enc, K.softmax, [this.#u(Lk, this.ctx, 0, 0), a.S], T * 16);
        gpu.dispatch(enc, K.attnOut, [this.#u(Lk, this.ctx, 0, 0), a.S, c.v, a.gate, a.att], T * 16);
      }
      this.#matmul(enc, p + 'attn_output.weight', a.att, a.y, T);
    }
    this.#norm(enc, a.y, this.w[p + 'post_attention_norm.weight'].buf, a.x, T, E, { mode: 1 });
    this.#norm(enc, a.x, this.w[p + 'ffn_norm.weight'].buf, a.h, T, E);
    this.#matmul(enc, p + 'ffn_gate.weight', a.h, a.g, T);
    this.#matmul(enc, p + 'ffn_up.weight', a.h, a.u, T);
    const n = T * F;
    gpu.dispatch(enc, K.swiglu, [this.#u(n, 0, 0, 0), a.g, a.u], [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)]);
    this.#matmul(enc, p + 'ffn_down.weight', a.g, a.y, T);
    this.#norm(enc, a.y, this.w[p + 'post_ffw_norm.weight'].buf, a.x, T, E, { mode: 1 });
  }

  /**
   * Top candidates of the last token's logits, computed on the GPU: { values, ids } (16 per 256 tokens of the
   * vocabulary). Much less to read back than every logit.
   */
  async candidates() {
    const gpu = this.gpu, n = Math.ceil(this.vocab / 256);
    this.cand ||= gpu.buffer(n * 16 * 8, 'activations');
    gpu.beginPlan('lfm3-topk');
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
