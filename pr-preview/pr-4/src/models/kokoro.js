// Kokoro-82M text to speech (StyleTTS2 prosody predictor + iSTFTNet decoder), from the official ggml-org GGUFs:
// the backbone (arch "kokoro", an ALBERT encoder over phonemes) and its mmproj (voices and everything after the
// encoder, tensors a.gen.*). A port of llama.cpp's src/models/kokoro.cpp and tools/mtmd/models/kokoro.cpp.
//
//   const k = await Kokoro.load(gpu, httpSource(backboneUrl), httpSource(mmprojUrl));
//   const pcm = await k.speak('həlˈoʊ wˈɜːld', { voice: 'af_heart' }); // Float32Array at 24 kHz
//
// Input is phonemes (misaki's output), one token per character; the G2P is not ported. Everything with weights
// runs on the GPU; the small signal steps (durations → alignment, the harmonic source and its STFT, the inverse
// STFT) run in JS between the stages, like speech.js does for LFM3's codec.

import { streamTensors } from '../loader.js';
import { readGGUF } from '../gguf.js';
import * as Kn from '../kernels.js';

export const KOKORO = { HIDDEN: 512, STYLE: 128, BINS: 11, RATE: 24000, UPS: 300, N_FFT: 20, HOP: 5, N_HARM: 9, MAX_PHONEMES: 510 };

const ew = (n) => [Math.min(Math.ceil(n / 256), 65535), Math.ceil(n / 256 / 65535)];
const SRC = {};
const mmSrc = (mt) => (SRC[mt] ||= Kn.matmulF32(mt));

function u(vals, floats = []) {
  const a = new Uint32Array(vals.length), f = new Float32Array(a.buffer);
  vals.forEach((v, i) => { if (floats.includes(i)) f[i] = v; else a[i] = v; });
  return a;
}

// ---- host-side signal steps (exported for the CPU reference and tests) ----

/** Seeded standard normal samples (mulberry32 + Box–Muller). */
export function gaussian(seed = 0) {
  let s = seed >>> 0, spare = null;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const r = Math.sqrt(-2 * Math.log(1 - next())), a = 2 * Math.PI * next();
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
}

const roundHalfEven = (x) => { const r = Math.round(x); return Math.abs(x - Math.trunc(x)) === 0.5 ? 2 * Math.round(x / 2) : r; };

/** Durations from the duration logits [n][nBins] (bias included), and the token index of each frame. */
export function alignIndex(logits, n, nBins, speed = 1) {
  const durations = new Int32Array(n);
  let F = 0;
  for (let t = 0; t < n; t++) {
    let s = 0;
    for (let j = 0; j < nBins; j++) s += 1 / (1 + Math.exp(-logits[t * nBins + j]));
    durations[t] = Math.max(1, roundHalfEven(Math.fround(s / speed)));
    F += durations[t];
  }
  const index = new Uint32Array(F);
  for (let t = 0, f = 0; t < n; t++) for (let r = 0; r < durations[t]; r++) index[f++] = t;
  return { durations, index };
}

const hann = (i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / KOKORO.N_FFT);

/**
 * hn-nsf harmonic source from the F0 curve [2·frames], then its STFT: rows [nHar][22] of 11 magnitudes and 11 phases.
 * `gauss` gives the noise samples.
 */
export function harSource(f0, linW, linB, gauss) {
  const { UPS, N_HARM: H, RATE, N_FFT, HOP } = KOKORO;
  const n0 = f0.length, nWav = n0 * UPS;
  const phase = new Float64Array(n0 * H);
  for (let h = 0; h < H; h++) {
    let cum = 0;
    for (let i = 0; i < n0; i++) {
      let rad = (f0[i] * (h + 1)) / RATE;
      rad -= Math.floor(rad);
      cum += rad;
      phase[i * H + h] = cum * 2 * Math.PI * UPS;
    }
  }
  const src = new Float32Array(nWav);
  for (let t = 0; t < nWav; t++) {
    const pos = Math.max(0, (t + 0.5) / UPS - 0.5);
    const i0 = Math.min(Math.floor(pos), n0 - 1), i1 = Math.min(i0 + 1, n0 - 1), frac = pos - i0;
    const uv = f0[Math.floor(t / UPS)] > 10 ? 1 : 0;
    const namp = uv * 0.003 + ((1 - uv) * 0.1) / 3;
    let merged = linB[0];
    for (let h = 0; h < H; h++) {
      const ph = (1 - frac) * phase[i0 * H + h] + frac * phase[i1 * H + h];
      merged += linW[h] * (0.1 * Math.sin(ph) * uv + namp * gauss());
    }
    src[t] = Math.tanh(merged);
  }
  const nb = N_FFT / 2 + 1, nHar = Math.floor(nWav / HOP) + 1, har = new Float32Array(nHar * 2 * nb);
  const frame = new Float64Array(N_FFT);
  for (let fr = 0; fr < nHar; fr++) {
    for (let i = 0; i < N_FFT; i++) {
      let idx = fr * HOP - N_FFT / 2 + i;
      if (idx < 0) idx = -idx;
      if (idx >= nWav) idx = 2 * (nWav - 1) - idx;
      frame[i] = src[idx] * hann(i);
    }
    for (let k = 0; k < nb; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < N_FFT; i++) { const a = (2 * Math.PI * ((k * i) % N_FFT)) / N_FFT; re += frame[i] * Math.cos(a); im -= frame[i] * Math.sin(a); }
      har[fr * 2 * nb + k] = Math.hypot(re, im);
      har[fr * 2 * nb + nb + k] = Math.atan2(im, re);
    }
  }
  return { har, nHar };
}

/** conv_post rows [frames][22] → waveform: magnitude exp(x), phase sin(x), Hann overlap-add, centering dropped. */
export function istft(spec, frames) {
  const { N_FFT, HOP } = KOKORO, nb = N_FFT / 2 + 1, nOut = Math.max(0, (frames - 1) * HOP);
  const acc = new Float64Array(nOut + N_FFT), wsum = new Float64Array(nOut + N_FFT);
  const cos = Float64Array.from({ length: N_FFT }, (_, i) => Math.cos((2 * Math.PI * i) / N_FFT));
  const sin = Float64Array.from({ length: N_FFT }, (_, i) => Math.sin((2 * Math.PI * i) / N_FFT));
  const re = new Float64Array(nb), im = new Float64Array(nb);
  for (let fr = 0; fr < frames; fr++) {
    const o = fr * 2 * nb;
    for (let k = 0; k < nb; k++) { const m = Math.exp(spec[o + k]), ph = Math.sin(spec[o + nb + k]); re[k] = m * Math.cos(ph); im[k] = m * Math.sin(ph); }
    for (let i = 0; i < N_FFT; i++) {
      let v = 0;
      for (let k = 0; k < nb; k++) { const a = (k * i) % N_FFT; v += (k === 0 || k === nb - 1 ? 1 : 2) * (re[k] * cos[a] - im[k] * sin[a]); }
      const w = hann(i);
      acc[fr * HOP + i] += (v / N_FFT) * w;
      wsum[fr * HOP + i] += w * w;
    }
  }
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) { const j = i + N_FFT / 2; out[i] = wsum[j] > 1e-11 ? acc[j] / wsum[j] : 0; }
  return out;
}

const tr = (a, rows, cols) => { const o = new Float32Array(a.length); for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) o[c * rows + r] = a[r * cols + c]; return o; };

// ---- the model ----

export class Kokoro {
  /**
   * Loads both GGUFs (all weights as f32, about 330 MB of GPU memory; the voices stay in the file and one 1 KB row
   * is read per utterance).
   * @param {import('../gpu.js').Gpu} gpu
   */
  static async load(gpu, backbone, mmproj, { onProgress } = {}) {
    const skip = (k) => k.startsWith('tokenizer.tts_dict') || k === 'tokenizer.chat_template' || k.startsWith('tokenizer.chat_template.');
    const [g1, g2] = await Promise.all([readGGUF(backbone, { skip }), readGGUF(mmproj, { skip })]);
    if (g1.kv['general.architecture'] !== 'kokoro') throw new Error('not a Kokoro backbone GGUF');
    const voices = g2.tensors.filter((t) => t.name.startsWith('voice.'));
    const gen = g2.tensors.filter((t) => t.name.startsWith('a.gen.'));
    if (!gen.length) throw new Error('this mmproj has no Kokoro generator');
    const total = g1.tensors.reduce((s, t) => s + t.nbytes, 0) + gen.reduce((s, t) => s + t.nbytes, 0);
    const prog = (base) => onProgress && ((p) => onProgress({ ...p, loaded: base + p.loaded, total }));
    const k = new Kokoro(gpu, g1.kv, g2.kv);
    const w1 = await streamTensors(gpu, backbone, g1.tensors, { tag: 'kokoro', onProgress: prog(0) });
    const w2 = await streamTensors(gpu, mmproj, gen, { tag: 'kokoro', onProgress: prog(g1.tensors.reduce((s, t) => s + t.nbytes, 0)) });
    for (const t of [...g1.tensors, ...gen]) k.w[t.name] = { buf: (w1[t.name] || w2[t.name]).buf, dims: t.dims };
    k.voiceInfo = Object.fromEntries(voices.map((t) => [t.name.slice(6), t]));
    k.voiceSrc = mmproj;
    await k.#init();
    return k;
  }

  /**
   * From weights in memory, for tests: `weights` maps tensor names to { dims (ggml order), data: Float32Array },
   * `voices` maps voice names to their tables ([rows][256]).
   */
  static async fromWeights(gpu, weights, kv, voices = {}) {
    const k = new Kokoro(gpu, kv, kv);
    for (const [name, { dims, data }] of Object.entries(weights)) {
      const buf = gpu.buffer(data.byteLength, 'kokoro');
      gpu.device.queue.writeBuffer(buf, 0, data);
      k.w[name] = { buf, dims };
    }
    k.voiceTables = voices;
    await k.#init();
    return k;
  }

  constructor(gpu, kv, genKv) {
    this.gpu = gpu;
    this.w = {};
    this.nLayer = kv['kokoro.block_count'] ?? 12;
    this.nHead = kv['kokoro.attention.head_count'] ?? 12;
    this.eps = kv['kokoro.attention.layer_norm_epsilon'] ?? 1e-12;
    const tokens = kv['tokenizer.ggml.tokens'] || [], types = kv['tokenizer.ggml.token_type'] || [];
    this.pad = kv['tokenizer.ggml.bos_token_id'] ?? 0;
    this.vocab = new Map();
    tokens.forEach((s, i) => { if ((types[i] ?? 1) === 1) this.vocab.set(s, i); });
    this.defaultVoice = genKv['clip.gen.audio.default_voice'] || 'af_heart';
    this.slots = new Map();
    this.trash = [];
    /** Milliseconds per stage of the last utterance. */
    this.times = {};
  }

  /** Voice names. */
  get voices() { return Object.keys(this.voiceInfo || this.voiceTables || {}); }

  has(name) { return name in this.w; }

  async #init() {
    const gpu = this.gpu;
    this.zero = gpu.buffer(16, 'kokoro');
    // W_hh is used transposed by the LSTM kernel, so its reads are coalesced.
    await gpu.run((enc) => {
      for (const [name, r] of Object.entries(this.w)) {
        if (!/\.weight_hh_l0(_reverse)?$/.test(name)) continue;
        const [cols, rows] = r.dims, t = gpu.buffer(rows * cols * 4, 'kokoro');
        gpu.dispatch(enc, Kn.transpose, [u([rows, cols, 0, 0]), r.buf, t], [Math.ceil(cols / 16), Math.ceil(rows / 16)]);
        r.old = r.buf; r.buf = t;
      }
    });
    for (const r of Object.values(this.w)) if (r.old) { gpu.free(r.old); delete r.old; }
  }

  /** Phoneme string → token ids, padded at both ends; unknown characters are dropped. */
  tokenize(phonemes) {
    const ids = [this.pad];
    for (const ch of phonemes) { const id = this.vocab.get(ch); if (id !== undefined) ids.push(id); }
    if (ids.length === 1) throw new Error('no phoneme the model knows');
    if (ids.length - 1 > KOKORO.MAX_PHONEMES) throw new Error(`at most ${KOKORO.MAX_PHONEMES} phonemes per call`);
    ids.push(this.pad);
    return ids;
  }

  /** The style row for an utterance of `len` phonemes: decoder style, then predictor style. */
  async voiceRow(voice, len) {
    if (this.voiceTables) {
      const tab = this.voiceTables[voice];
      if (!tab) throw new Error(`unknown voice "${voice}"`);
      const row = Math.min(len - 1, tab.length / 256 - 1);
      return tab.slice(row * 256, row * 256 + 256);
    }
    const t = this.voiceInfo[voice];
    if (!t) throw new Error(`unknown voice "${voice}"`);
    if (t.typeName !== 'F32') throw new Error(`voice ${voice}: ${t.typeName} voices are not supported`);
    const row = Math.min(len - 1, t.dims[1] - 1);
    const b = await this.voiceSrc.read(t.offset + row * 1024, 1024);
    return new Float32Array(b.slice().buffer);
  }

  /** Speaks a phoneme string. Returns 24 kHz PCM. */
  async speak(phonemes, { voice = this.defaultVoice, speed = 1, seed = 0 } = {}) {
    const ids = this.tokenize(phonemes);
    const row = await this.voiceRow(voice, ids.length - 2);
    return (await this.synth(ids, row, { speed, seed })).audio;
  }

  // A scratch buffer by name, grown when too small (the old one is freed after the utterance).
  s(name, floats) {
    let b = this.slots.get(name);
    if (!b || b.size < floats * 4) {
      if (b) this.trash.push(b);
      b = this.gpu.buffer(Math.max(16, floats * 4), 'kokoro-act');
      this.slots.set(name, b);
    }
    return b;
  }

  /** Frees the weights and scratch buffers. */
  dispose() {
    for (const r of Object.values(this.w)) this.gpu.free(r.buf);
    for (const b of [...this.slots.values(), ...this.trash]) this.gpu.free(b);
    this.gpu.free(this.zero);
    this.w = {}; this.slots.clear(); this.trash = [];
  }

  upload(data, name) {
    const b = this.s(name, data.length);
    this.gpu.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  // ---- dispatch helpers (record into `enc`) ----

  mm(enc, name, X, Y, T) {
    return this.mmW(enc, name + '.weight', this.has(name + '.bias') ? this.w[name + '.bias'].buf : null, X, Y, T);
  }

  mmW(enc, wName, B, X, Y, T) {
    const W = this.w[wName], [K, rows] = W.dims, mt = T === 1 ? 1 : Kn.MAXT;
    this.gpu.dispatch(enc, mmSrc(mt), [u([rows, K, T, B ? 1 : 0]), W.buf, X, Y, B || this.zero], [Math.min(rows, 32768), Math.ceil(rows / 32768), Math.ceil(T / mt)]);
    return rows;
  }

  conv(enc, name, X, T, Y, { s = 1, p = 0, d = 1, xOff = 0, yOff = 0, alpha = 1, beta = 0, slope = 1 } = {}) {
    const W = this.w[name + '.weight'], [K, IC, OC] = W.dims;
    const L = Math.floor((T + 2 * p - d * (K - 1) - 1) / s) + 1;
    const hasB = this.has(name + '.bias'), B = hasB ? this.w[name + '.bias'].buf : this.zero;
    this.gpu.dispatch(enc, Kn.conv1d, [u([OC, IC, K, T, L, s, p, d, xOff, yOff, hasB ? 1 : 0, 0, alpha, beta, slope, 0], [12, 13, 14]), W.buf, X, B, Y], [Math.ceil(L / 64), Math.ceil(OC / 64)]);
    return { L, C: OC };
  }

  adain(enc, name, X, C, T, Y, style, mode, slope = 1, alpha = null) {
    const G = this.s('gb', 2 * C);
    this.mm(enc, name + '.fc', style, G, 1);
    this.gpu.dispatch(enc, Kn.adain, [u([C, T, mode, slope, 1e-5, 0, 0, 0], [3, 4]), X, G, alpha ? this.w[alpha].buf : this.zero, Y], [C]);
  }

  axpby(enc, X, Y, n, a, b, xOff = 0, yOff = 0) {
    this.gpu.dispatch(enc, Kn.axpby, [u([n, xOff, yOff, a, b, 0, 0, 0], [3, 4]), X, Y], ew(n));
  }

  transpose(enc, X, Y, rows, cols) {
    this.gpu.dispatch(enc, Kn.transpose, [u([rows, cols, 0, 0]), X, Y], [Math.ceil(cols / 16), Math.ceil(rows / 16)]);
  }

  gather(enc, W, I, Y, K, n) {
    this.gpu.dispatch(enc, Kn.gatherF32, [u([K, n, 0, 0]), W, I, Y], [Math.ceil((n * K) / 64)]);
  }

  layernorm(enc, X, name, Y, rows, N, eps) {
    this.gpu.dispatch(enc, Kn.layernorm, [u([N, eps, 0, 0], [1]), X, this.w[name + '.weight'].buf, this.w[name + '.bias'].buf, Y], [rows]);
  }

  /** Bidirectional LSTM on token-major X [n][nIn] → Y [n][512]; the biases ride on the input projections. */
  lstm(enc, name, X, n, Y) {
    const G0 = this.s('lstm-g0', n * 1024), G1 = this.s('lstm-g1', n * 1024);
    this.mmW(enc, name + '.weight_ih_l0', this.w[name + '.bias_l0'].buf, X, G0, n);
    this.mmW(enc, name + '.weight_ih_l0_reverse', this.w[name + '.bias_l0_reverse'].buf, X, G1, n);
    this.gpu.dispatch(enc, Kn.lstm, [u([n, 0, 0, 0]), G0, G1, this.w[name + '.weight_hh_l0'].buf, this.w[name + '.weight_hh_l0_reverse'].buf, Y], [2]);
  }

  /** Kokoro AdainResBlk1d on X [C][T] into Y (at yOff). Returns the output { C, T }. */
  resblk(enc, name, X, C, T, style, Y, yOff = 0) {
    const up = this.has(name + '.pool.weight'), T2 = up ? 2 * T : T, r = this.s('rb-r', C * T);
    this.adain(enc, name + '.norm1', X, C, T, r, style, 1, 0.2);
    let src = r;
    if (up) {
      src = this.s('rb-u', C * T2);
      this.gpu.dispatch(enc, Kn.up2, [u([C, T, 0, 0]), r, this.w[name + '.pool.weight'].buf, this.w[name + '.pool.bias'].buf, src], [Math.ceil(T2 / 64), C]);
    }
    const C2 = this.w[name + '.conv1.weight'].dims[2], h = this.s('rb-h', C2 * T2), h2 = this.s('rb-h2', C2 * T2);
    this.conv(enc, name + '.conv1', src, T2, h, { p: 1 });
    this.adain(enc, name + '.norm2', h, C2, T2, h2, style, 1, 0.2);
    this.conv(enc, name + '.conv2', h2, T2, Y, { p: 1, yOff });
    let sc = X;
    if (up) {
      sc = this.s('rb-u2', C * T2);
      this.gpu.dispatch(enc, Kn.up2, [u([C, T, 1, 0]), X, this.zero, this.zero, sc], [Math.ceil(T2 / 64), C]);
    }
    if (this.has(name + '.conv1x1.weight')) this.conv(enc, name + '.conv1x1', sc, T2, Y, { yOff, alpha: Math.SQRT1_2, beta: 1 });
    else this.axpby(enc, sc, Y, C2 * T2, Math.SQRT1_2, Math.SQRT1_2, 0, yOff);
    return { C: C2, T: T2 };
  }

  /** Generator AdaINResBlock1 in place on X [C][T]; `tag` keeps the scratch of each stage apart. */
  genResblock(enc, name, X, C, T, style, tag) {
    const dil = [1, 3, 5], t1 = this.s('g1' + tag, C * T), t2 = this.s('g2' + tag, C * T);
    for (let j = 0; j < 3; j++) {
      const k = this.w[`${name}.convs1.${j}.weight`].dims[0];
      this.adain(enc, `${name}.adain1.${j}`, X, C, T, t1, style, 2, 1, `${name}.alpha1.${j}`);
      this.conv(enc, `${name}.convs1.${j}`, t1, T, t2, { p: (k * dil[j] - dil[j]) / 2, d: dil[j] });
      this.adain(enc, `${name}.adain2.${j}`, t2, C, T, t1, style, 2, 1, `${name}.alpha2.${j}`);
      this.conv(enc, `${name}.convs2.${j}`, t1, T, X, { p: (k - 1) / 2, beta: 1 });
    }
  }

  /**
   * One utterance from padded token ids and a voice row (256 values). Returns { audio, durations, f0, spec }.
   */
  async synth(tokens, voiceRow, { speed = 1, seed = 0 } = {}) {
    const gpu = this.gpu, n = tokens.length, Hd = KOKORO.HIDDEN, S = KOKORO.STYLE, nCat = Hd + S;
    const times = (this.times = {});
    const run = async (label, f) => { const t0 = performance.now(); await gpu.run(f); times[label] = (times[label] || 0) + Math.round(performance.now() - t0); };
    const ids = this.upload(Uint32Array.from(tokens), 'ids');
    const sDec = this.upload(voiceRow.slice(0, S), 'style-dec'), sPred = this.upload(voiceRow.slice(S, 2 * S), 'style-pred');

    // 1. ALBERT over the phonemes → bert [n][768]
    const E = this.w['token_embd.weight'].dims[0], e = this.w['position_embd.weight'].dims[0], H = this.nHead, D = E / H;
    const Ff = this.w['blk.0.ffn_up.weight'].dims[1];
    const x = this.s('bert', n * E), a = this.s('bert-a', n * e), b = this.s('bert-b', n * e);
    const q = this.s('q', n * E), kk = this.s('k', n * E), v = this.s('v', n * E), att = this.s('att', n * E), o = this.s('o', n * E);
    const sc = this.s('scores', n * n * H), ff = this.s('ff', n * Ff);
    // 2. text CNN → text [n][512]
    const tg = this.s('t-g', n * Hd), ta = this.s('t-a', n * Hd), tb = this.s('t-b', n * Hd), text = this.s('text', n * Hd);
    // 3. prosody up to the durations
    const be = this.s('be', n * Hd), px = this.s('px', n * nCat), ly = this.s('ly', n * Hd), gb = this.s('ada', 2 * Hd);
    const pw = this.w['a.gen.pred.duration_proj.linear_layer.weight'], nBins = pw.dims[1], logits = this.s('logits', n * nBins);
    await run('encoder', (enc) => {
      gpu.dispatch(enc, Kn.embedAdd, [u([E, e, n, 0]), this.w['token_embd.weight'].buf, this.w['position_embd.weight'].buf, ids, a], [Math.ceil((n * e) / 64)]);
      this.layernorm(enc, a, 'token_embd_norm', b, n, e, this.eps);
      this.mm(enc, 'token_embd_proj', b, x, n);
      for (let l = 0; l < this.nLayer; l++) {
        this.mm(enc, 'blk.0.attn_q', x, q, n); this.mm(enc, 'blk.0.attn_k', x, kk, n); this.mm(enc, 'blk.0.attn_v', x, v, n);
        gpu.dispatch(enc, Kn.encScores, [u([n, 0, H, H, D, 1 / Math.sqrt(D), 0, 0], [5]), q, kk, sc], [Math.ceil(n / 64), n * H]);
        gpu.dispatch(enc, Kn.softmax, [u([n, n, 0, 0]), sc], [n * H]);
        gpu.dispatch(enc, Kn.encOut, [u([n, 0, H, H, D, 0, 0, 0]), sc, v, att], [n * H]);
        this.mm(enc, 'blk.0.attn_output', att, o, n);
        gpu.dispatch(enc, Kn.addInto, [u([n * E, 0, 0, 0]), x, o], ew(n * E));
        this.layernorm(enc, o, 'blk.0.attn_output_norm', x, n, E, this.eps);
        this.mm(enc, 'blk.0.ffn_up', x, ff, n);
        gpu.dispatch(enc, Kn.geluTanh, [u([n * Ff, 0, 0, 0]), ff], ew(n * Ff));
        this.mm(enc, 'blk.0.ffn_down', ff, o, n);
        gpu.dispatch(enc, Kn.addInto, [u([n * E, 0, 0, 0]), x, o], ew(n * E));
        this.layernorm(enc, o, 'blk.0.layer_output_norm', x, n, E, this.eps);
      }
      // text encoder CNN
      this.gather(enc, this.w['a.gen.text.embedding.weight'].buf, ids, tg, Hd, n);
      this.transpose(enc, tg, ta, n, Hd);
      for (let i = 0; this.has(`a.gen.text.cnn.${i}.0.weight`); i++) {
        const p = `a.gen.text.cnn.${i}`, K = this.w[p + '.0.weight'].dims[0];
        this.conv(enc, p + '.0', ta, n, tb, { p: (K - 1) / 2 });
        gpu.dispatch(enc, Kn.lnChanLeaky, [u([Hd, n, 0, 0]), tb, this.w[p + '.1.gamma'].buf, this.w[p + '.1.beta'].buf], [Math.ceil(n / 64)]);
        this.axpby(enc, tb, ta, Hd * n, 0, 1);
      }
      this.transpose(enc, ta, text, Hd, n);
      // prosody: bert_enc, then 3 × (LSTM, AdaLayerNorm), then the duration LSTM and projection
      this.mm(enc, 'a.gen.bert_enc', x, be, n);
      gpu.dispatch(enc, Kn.prosodyRows, [u([Hd, S, 0, 0]), be, sPred, px], [n]);
      for (let il = 0; this.has(`a.gen.pred.text_encoder.lstms.${2 * il}.weight_ih_l0`); il++) {
        this.lstm(enc, `a.gen.pred.text_encoder.lstms.${2 * il}`, px, n, ly);
        this.mm(enc, `a.gen.pred.text_encoder.lstms.${2 * il + 1}.fc`, sPred, gb, 1);
        gpu.dispatch(enc, Kn.prosodyRows, [u([Hd, S, 1, 0]), ly, gb, px], [n]);
      }
      this.lstm(enc, 'a.gen.pred.lstm', px, n, ly);
      this.mm(enc, 'a.gen.pred.duration_proj.linear_layer', ly, logits, n);
    });
    const lg = new Float32Array(await gpu.read(logits, n * nBins * 4));
    const { durations, index } = alignIndex(lg, n, nBins, speed);
    const F = index.length;

    // 4. aligned LSTMs → shared [512][F], asr [512][F]
    const idx = this.upload(index, 'index'), pa = this.s('pa', F * nCat), sh = this.s('sh', F * Hd), shT = this.s('shT', F * Hd);
    const tl = this.s('tl', n * Hd), ar = this.s('ar', F * Hd), asrT = this.s('asrT', F * Hd);
    await run('prosody', (enc) => {
      this.gather(enc, px, idx, pa, nCat, F);
      this.lstm(enc, 'a.gen.pred.shared', pa, F, sh);
      this.transpose(enc, sh, shT, F, Hd);
      this.lstm(enc, 'a.gen.text.lstm', text, n, tl);
      this.gather(enc, tl, idx, ar, Hd, F);
      this.transpose(enc, ar, asrT, F, Hd);
    });

    // 5. F0 and energy curves [2F]
    const f0b = this.s('f0', 2 * F), enb = this.s('en', 2 * F);
    for (const [curve, out] of [['F0', f0b], ['N', enb]]) {
      let cur = { buf: shT, C: Hd, T: F };
      for (let i = 0; i < 3; i++) {
        const dst = this.s(`fn${i % 2}`, 512 * 2 * F);
        await run('f0n', (enc) => { const r = this.resblk(enc, `a.gen.pred.${curve}.${i}`, cur.buf, cur.C, cur.T, sPred, dst); cur = { buf: dst, ...r }; });
      }
      await run('f0n', (enc) => this.conv(enc, `a.gen.pred.${curve}_proj`, cur.buf, cur.T, out));
    }
    const f0 = new Float32Array(await gpu.read(f0b, 2 * F * 4));

    // 6. decoder: encode, then 4 decode blocks over [x, asr_res, f0, energy]
    const p = 'a.gen.dec.', x514 = this.s('x514', 514 * F), catA = this.s('catA', 1090 * F), catB = this.s('catB', 1090 * F);
    await run('decoder', (enc) => {
      this.axpby(enc, asrT, x514, Hd * F, 0, 1);
      this.conv(enc, p + 'F0_conv', f0b, 2 * F, x514, { s: 2, p: 1, yOff: Hd * F });
      this.conv(enc, p + 'N_conv', enb, 2 * F, x514, { s: 2, p: 1, yOff: (Hd + 1) * F });
      for (const cat of [catA, catB]) {
        this.conv(enc, p + 'asr_res.0', asrT, F, cat, { yOff: 1024 * F });
        this.axpby(enc, x514, cat, 2 * F, 0, 1, Hd * F, 1088 * F);
      }
      this.resblk(enc, p + 'encode', x514, Hd + 2, F, sDec, catA);
    });
    let cur = { buf: catA, C: 1090, T: F };
    for (let i = 0; this.has(`${p}decode.${i}.conv1.weight`); i++) {
      const last = !this.has(`${p}decode.${i + 1}.conv1.weight`);
      const dst = last ? this.s('d-last', 512 * 2 * F) : cur.buf === catA ? catB : catA;
      // every block reads the whole concat (1090 channels); its output fills the first 1024
      await run('decoder', (enc) => { const r = this.resblk(enc, `${p}decode.${i}`, cur.buf, 1090, F, sDec, dst); cur = { buf: dst, ...r }; });
    }

    // 7. harmonic source (JS), then the generator
    const t0 = performance.now();
    const { har, nHar } = harSource(f0, await this.host(p + 'generator.m_source.l_linear.weight'), await this.host(p + 'generator.m_source.l_linear.bias'), gaussian(seed));
    const harT = this.upload(tr(har, nHar, 2 * KOKORO.BINS), 'harT');
    times.source = Math.round(performance.now() - t0);
    const g = p + 'generator.';
    for (let i = 0; this.has(`${g}ups.${i}.weight`); i++) {
      const lastUp = !this.has(`${g}ups.${i + 1}.weight`);
      const sw = this.w[`${g}noise_convs.${i}.weight`].dims, sk = sw[0], ss = sk > 1 ? sk / 2 : 1, Cs = sw[2];
      const Ls = Math.floor((nHar + 2 * (sk > 1 ? (ss + 1) >> 1 : 0) - (sk - 1) - 1) / ss) + 1;
      const src = this.s(`src${i}`, Cs * Ls);
      const uw = this.w[`${g}ups.${i}.weight`], [K, OC, IC] = uw.dims, stride = K / 2, Lo = cur.T * stride + (lastUp ? 1 : 0);
      if (Lo !== Ls) throw new Error(`kokoro: generator stage ${i} lengths ${Lo} and ${Ls} differ`);
      const y = this.s(`y${i}`, OC * Lo), tmp = this.s(`tmp${i}`, OC * Lo), acc = this.s(`acc${i}`, OC * Lo);
      await run('generator', (enc) => this.conv(enc, `${g}noise_convs.${i}`, harT, nHar, src, { s: ss, p: sk > 1 ? (ss + 1) >> 1 : 0 }));
      await run('generator', (enc) => this.genResblock(enc, `${g}noise_res.${i}`, src, Cs, Ls, sDec, `s${i}`));
      await run('generator', (enc) => {
        gpu.dispatch(enc, Kn.convT1d, [u([IC, OC, K, cur.T, stride, stride / 2, Lo, lastUp ? 1 : 0, 0.1, 0, 0, 0], [8]), uw.buf, cur.buf, this.w[`${g}ups.${i}.bias`].buf, y], [Math.ceil(Lo / 64), OC]);
        this.axpby(enc, src, y, OC * Lo, 1, 1);
      });
      for (let j = 0; j < 3; j++) {
        await run('generator', (enc) => {
          this.axpby(enc, y, tmp, OC * Lo, 0, 1);
          this.genResblock(enc, `${g}resblocks.${i * 3 + j}`, tmp, OC, Lo, sDec, `r${i}`);
          this.axpby(enc, tmp, acc, OC * Lo, j ? 1 : 0, 1 / 3);
        });
      }
      cur = { buf: acc, C: OC, T: Lo };
    }
    const specB = this.s('spec', 2 * KOKORO.BINS * cur.T);
    await run('generator', (enc) => this.conv(enc, g + 'conv_post', cur.buf, cur.T, specB, { p: 3, slope: 0.01 }));
    const specCM = new Float32Array(await gpu.read(specB, 2 * KOKORO.BINS * cur.T * 4));
    const t1 = performance.now();
    const spec = tr(specCM, 2 * KOKORO.BINS, cur.T);
    const audio = istft(spec, cur.T);
    times.istft = Math.round(performance.now() - t1);
    for (const old of this.trash.splice(0)) gpu.free(old);
    return { audio, durations, f0, spec };
  }

  // Small tensors read back once (the harmonic source mix).
  async host(name) {
    this.hostCache ||= {};
    if (!this.hostCache[name]) { const r = this.w[name]; this.hostCache[name] = new Float32Array(await this.gpu.read(r.buf, r.dims.reduce((a, b) => a * b, 1) * 4)); }
    return this.hostCache[name];
  }
}

