// Speech output helpers: streaming inverse STFT (llama.cpp's mtmd_audio_streaming_istft), sampling of codebook
// codes, and WAV encoding.

export const SPEECH_RATE = 24000;

/** Streaming ISTFT: periodic Hann window, overlap-add normalized by the window sum, centered padding removed. */
export class Istft {
  constructor(nFft = 1280, hop = 320) {
    this.n = nFft; this.hop = hop; this.bins = nFft / 2 + 1;
    this.win = Float32Array.from({ length: nFft }, (_, i) => 0.5 * (1 - Math.cos((2 * Math.PI * i) / nFft)));
    this.cos = Float32Array.from({ length: nFft }, (_, i) => Math.cos((2 * Math.PI * i) / nFft));
    this.sin = Float32Array.from({ length: nFft }, (_, i) => Math.sin((2 * Math.PI * i) / nFft));
    this.reset();
  }
  reset() {
    this.overlap = new Float32Array(this.n); this.wsum = new Float32Array(this.n);
    this.pad = (this.n - this.hop) / 2;
  }
  /** One frame of 641 log magnitudes followed by 641 phases → up to `hop` samples. */
  frame(logMagPhase, off = 0) {
    const n = this.n, bins = this.bins, re = new Float32Array(bins), im = new Float32Array(bins);
    for (let k = 0; k < bins; k++) {
      const m = Math.exp(logMagPhase[off + k]), ph = logMagPhase[off + bins + k];
      re[k] = m * Math.cos(ph); im[k] = m * Math.sin(ph);
    }
    // Real inverse DFT of a conjugate-symmetric spectrum (bins 0 and n/2 contribute their real parts only).
    const x = new Float32Array(n);
    for (let t = 0; t < n; t++) {
      let s = re[0] + (t & 1 ? -re[bins - 1] : re[bins - 1]);
      let q = t;
      for (let k = 1; k < bins - 1; k++) { s += 2 * (re[k] * this.cos[q] - im[k] * this.sin[q]); q += t; if (q >= n) q %= n; }
      x[t] = s / n;
    }
    for (let j = 0; j < n; j++) { this.wsum[j] += this.win[j] * this.win[j]; this.overlap[j] += x[j] * this.win[j]; }
    const out = this.#take(this.hop);
    const drop = Math.min(this.pad, out.length);
    this.pad -= drop;
    return out.subarray(drop);
  }
  #take(m) {
    const out = new Float32Array(m);
    for (let i = 0; i < m; i++) out[i] = this.wsum[i] > 1e-8 ? this.overlap[i] / this.wsum[i] : this.overlap[i];
    this.overlap.copyWithin(0, m); this.overlap.fill(0, this.n - m);
    this.wsum.copyWithin(0, m); this.wsum.fill(0, this.n - m);
    return out;
  }
  /** The tail after the last frame, minus the centered padding. */
  flush() {
    const parts = [];
    for (let left = this.n - this.hop; left > 0; left -= this.hop) parts.push(this.#take(Math.min(left, this.hop)));
    const all = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
    let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
    const trim = (this.n - this.hop) / 2;
    this.reset();
    return all.subarray(0, Math.max(0, all.length - trim));
  }
}

/**
 * Picks one codebook's code from its logits: CTRL repeat penalty over recent codes, then temperature sampling
 * (temp 0 = greedy).
 */
export function sampleCode(logits, recent, { temp, penalty }) {
  const l = Float32Array.from(logits);
  if (penalty > 1) for (const c of new Set(recent)) l[c] = l[c] > 0 ? l[c] / penalty : l[c] * penalty;
  let best = 0;
  for (let i = 1; i < l.length; i++) if (l[i] > l[best]) best = i;
  if (!temp) return best;
  let sum = 0;
  const p = new Float64Array(l.length);
  for (let i = 0; i < l.length; i++) { p[i] = Math.exp((l[i] - l[best]) / temp); sum += p[i]; }
  let r = Math.random() * sum;
  for (let i = 0; i < l.length; i++) { r -= p[i]; if (r <= 0) return i; }
  return l.length - 1;
}

/** Float32 PCM → 16-bit mono WAV bytes. */
export function wav(pcm, rate = SPEECH_RATE) {
  const b = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const s = (o, t) => [...t].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  s(0, 'RIFF'); b.setUint32(4, 36 + pcm.length * 2, true); s(8, 'WAVEfmt '); b.setUint32(16, 16, true);
  b.setUint16(20, 1, true); b.setUint16(22, 1, true); b.setUint32(24, rate, true); b.setUint32(28, rate * 2, true);
  b.setUint16(32, 2, true); b.setUint16(34, 16, true); s(36, 'data'); b.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) b.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(pcm[i] * 32767))), true);
  return new Uint8Array(b.buffer);
}

/**
 * Cuts the silence a TTS clip is padded with, keeping `before` s ahead of the speech and `after` s behind it (a
 * natural pause, with room for a word that fades out quietly). Kokoro pads each clip with ~0.3 s and ~0.5 s.
 * shortcut: the threshold is a fixed fraction of the clip's peak; revisit if quiet voices lose their first sound.
 * @param {Float32Array} pcm
 */
export function trimSilence(pcm, { rate = SPEECH_RATE, before = 0.03, after = 0.25, floor = 0.01 } = {}) {
  const thr = floor * pcm.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  let a = 0, b = pcm.length - 1;
  while (a < b && Math.abs(pcm[a]) < thr) a++;
  while (b > a && Math.abs(pcm[b]) < thr) b--;
  return pcm.subarray(Math.max(0, a - Math.round(before * rate)), Math.min(pcm.length, b + 1 + Math.round(after * rate)));
}

/**
 * Groups sentences into chunks to speak one per call, so the voice carries its intonation across sentences and
 * there are fewer hand-offs. The first chunk is one sentence (fast first sound); each later one may be up to
 * `grow`× the one before it (synthesis runs ~4× faster than playback on an iPhone, so it's ready before that one
 * ends), and at most `max` long. A sentence longer than the cap still goes alone.
 * @param {string[]} sentences
 */
export function chunkSentences(sentences, { max = 250, grow = 3 } = {}) {
  const out = [];
  let prev = 0;
  for (const s of sentences) {
    const last = out.at(-1);
    if (last && out.length > 1 && last.length + 1 + s.length <= Math.min(max, grow * prev)) out[out.length - 1] = last + ' ' + s;
    else { if (last) prev = last.length; out.push(s); }
  }
  return out;
}

/**
 * Plays clips back to back as they arrive, without gaps or cut endings. It uses <audio> elements, not Web
 * Audio, which stayed silent on an iPhone. There are three: one plays, the next clip loads into another, and the
 * third finishes the clip before. Switching elements takes ~110 ms on an iPhone, so each clip starts `lead` s
 * before the one before it ends, inside that clip's trailing silence (keep `after` in trimSilence longer).
 * Call `unlock()` inside a tap first.
 */
export class ClipPlayer {
  constructor({ rate = SPEECH_RATE, lead = 0.12 } = {}) {
    Object.assign(this, { rate, lead, queue: [], cur: -1, playing: false, started: 0, dur: 0, timer: 0 });
    /** ms each clip started past the end of the one before (negative = early, inside its trailing silence) */
    this.handoffs = [];
    this.els = Array.from({ length: 3 }, () => Object.assign(new Audio(), { preload: 'auto' }));
    for (const e of this.els) e.onended = () => { if (e === this.els[this.cur]) this.#next(); else if (!this.playing) this.#idle(); };
  }

  /** Starts every element on a moment of silence; iOS only lets an element play after one play() in a tap. */
  unlock() {
    const u = URL.createObjectURL(new Blob([wav(new Float32Array(240), this.rate)], { type: 'audio/wav' }));
    for (const e of this.els) { e.src = u; e.play().catch(() => {}); }
  }

  /** @param {Float32Array} pcm */
  push(pcm) {
    this.queue.push({ u: URL.createObjectURL(new Blob([wav(pcm, this.rate)], { type: 'audio/wav' })), dur: pcm.length / this.rate });
    if (!this.playing) this.#next();
    else if (this.queue.length === 1) this.#preload();
  }

  /** Resolves once everything pushed so far has played. */
  done() { return this.playing ? new Promise((r) => (this.onIdle = r)) : Promise.resolve(); }

  stop() {
    clearTimeout(this.timer);
    this.queue.length = 0;
    for (const e of this.els) e.pause();
    this.playing = false;
    this.#idle();
  }

  #idle() { const r = this.onIdle; this.onIdle = null; r?.(); }

  #preload() { if (this.queue[0]) { const e = this.els[(this.cur + 1) % 3]; e.src = this.queue[0].u; e.load(); } }

  #next() {
    clearTimeout(this.timer);
    const q = this.queue.shift();
    this.playing = !!q;
    if (!q) { this.started = 0; return this.#idle(); }
    this.cur = (this.cur + 1) % 3;
    const e = this.els[this.cur];
    if (e.src !== q.u) e.src = q.u;
    e.onplaying = () => {
      e.onplaying = null;
      const now = performance.now();
      if (this.started) this.handoffs.push(now - this.started - this.dur * 1000);
      this.started = now; this.dur = q.dur;
      this.timer = setTimeout(() => { if (this.queue[0]) this.#next(); }, Math.max(0, (q.dur - this.lead) * 1000));
    };
    e.play().catch(() => {});
    this.#preload();
  }
}
