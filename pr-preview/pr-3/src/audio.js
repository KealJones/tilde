// Audio input for LFM3 Omni: 16 kHz mono PCM → 128-bin log-mel features (llama.cpp's lfm3a preprocessor:
// 400-point DFT with a periodic Hann window, hop 160, Slaney mel filterbank with area norm, log10 floored at
// 1e-12, then (x + 4) / 4).

export const SAMPLE_RATE = 16000;
const N_FFT = 400, HOP = 160, N_MEL = 128, BINS = N_FFT / 2 + 1;

let tables = null;
function init() {
  if (tables) return tables;
  const cos = new Float32Array(N_FFT), sin = new Float32Array(N_FFT), hann = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) {
    const th = (2 * Math.PI * i) / N_FFT;
    cos[i] = Math.fround(Math.cos(th)); sin[i] = Math.fround(Math.sin(th));
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N_FFT));
  }
  // Slaney mel scale (librosa default), triangles normalized to unit area.
  const minLogHz = 1000, linSlope = 3 / 200, minLogMel = minLogHz * linSlope, logStep = Math.log(6.4) / 27;
  const hz2mel = (f) => (f < minLogHz ? f * linSlope : minLogMel + Math.log(f / minLogHz) / logStep);
  const mel2hz = (m) => (m < minLogMel ? m / linSlope : minLogHz * Math.exp((m - minLogMel) * logStep));
  const lo = hz2mel(0), hi = hz2mel(SAMPLE_RATE / 2);
  const hz = Array.from({ length: N_MEL + 2 }, (_, i) => mel2hz(lo + ((hi - lo) * i) / (N_MEL + 1)));
  const filters = new Float32Array(N_MEL * BINS);
  for (let m = 0; m < N_MEL; m++) {
    const [l, c, r] = [hz[m], hz[m + 1], hz[m + 2]];
    const enorm = 2 / Math.max(1e-30, r - l);
    for (let k = 0; k < BINS; k++) {
      const f = (k * SAMPLE_RATE) / N_FFT;
      let w = 0;
      if (f >= l && f <= c) w = (f - l) / Math.max(1e-30, c - l);
      else if (f > c && f <= r) w = (r - f) / Math.max(1e-30, r - c);
      filters[m * BINS + k] = w * enorm;
    }
  }
  return (tables = { cos, sin, hann, filters });
}

/**
 * Log-mel features, mel-major: Float32Array [128 × frames] (frame index fastest), plus the frame count.
 * @param {Float32Array} pcm 16 kHz mono samples in [-1, 1]
 */
export function logMel(pcm) {
  const { cos, sin, hann, filters } = init();
  // Pad on the right so the last frame keeps every sample.
  const n = pcm.length;
  const padded = n > N_FFT ? n + ((HOP - ((n - N_FFT) % HOP)) % HOP) : N_FFT;
  const x = new Float32Array(padded);
  x.set(pcm.subarray(0, Math.min(n, padded)));
  const frames = (padded - N_FFT) / HOP + 1;
  const out = new Float32Array(N_MEL * frames);
  const win = new Float32Array(N_FFT), power = new Float64Array(BINS);
  for (let t = 0; t < frames; t++) {
    for (let j = 0; j < N_FFT; j++) win[j] = x[t * HOP + j] * hann[j];
    for (let k = 0; k < BINS; k++) {
      let re = 0, im = 0, q = 0;
      for (let j = 0; j < N_FFT; j++) { re += win[j] * cos[q]; im += win[j] * sin[q]; q += k; if (q >= N_FFT) q -= N_FFT; }
      power[k] = re * re + im * im;
    }
    for (let m = 0; m < N_MEL; m++) {
      let s = 0;
      const row = m * BINS;
      for (let k = 0; k < BINS; k++) s += power[k] * filters[row + k];
      out[m * frames + t] = (Math.log10(Math.max(s, 1e-12)) + 4) / 4;
    }
  }
  return { mel: out, frames };
}

/** 16-bit PCM WAV bytes → Float32Array (first channel); the file must already be 16 kHz. */
export function parseWav(buf) {
  const dv = new DataView(buf instanceof ArrayBuffer ? buf : buf.buffer, buf.byteOffset || 0);
  let p = 12, rate = 0, ch = 1, bits = 16;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(p), dv.getUint8(p + 1), dv.getUint8(p + 2), dv.getUint8(p + 3));
    const size = dv.getUint32(p + 4, true);
    if (id === 'fmt ') { ch = dv.getUint16(p + 10, true); rate = dv.getUint32(p + 12, true); bits = dv.getUint16(p + 22, true); }
    if (id === 'data') {
      if (bits !== 16) throw new Error('only 16-bit WAV is supported');
      if (rate !== SAMPLE_RATE) throw new Error(`WAV is ${rate} Hz; use decodeAudio() to resample`);
      const n = Math.floor(size / 2 / ch), out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = dv.getInt16(p + 8 + i * 2 * ch, true) / 32768;
      return out;
    }
    p += 8 + size + (size & 1);
  }
  throw new Error('no data chunk in WAV');
}

/** Any audio the browser can decode (or a recorded Blob) → 16 kHz mono Float32Array. */
export async function decodeAudio(data) {
  const bytes = data instanceof Blob ? await data.arrayBuffer() : data;
  const ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(bytes.slice(0));
  const frames = Math.ceil(decoded.duration * SAMPLE_RATE);
  const off = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}
