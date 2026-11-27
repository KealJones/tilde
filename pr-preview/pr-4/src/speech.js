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
