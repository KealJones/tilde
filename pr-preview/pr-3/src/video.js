// Video input for LFM3-VL / Omni, the way llama.cpp samples it: 2 frames a second (4 to 384 frames for Omni),
// spread evenly over the source frames, read in pairs; Omni also hears the soundtrack, a slice after each pair.

import { smartResize } from './image.js';
import { decodeAudio } from './audio.js';

/**
 * Frame times and timestamps for a clip: n = clamp(⌊duration·fps⌋, minFrames, maxFrames) frames at
 * round(linspace(0, nSrc − 1, n)) of the source frames (llama.cpp mtmd_video_config_init).
 * @param {number} duration seconds
 * @param {{ fps?: number, srcFps?: number, minFrames?: number, maxFrames?: number }} [o]
 * @returns {{ times: number[], nSrc: number, srcFps: number }}
 */
export function frameTimes(duration, { fps = 2, srcFps = 30, minFrames = 4, maxFrames = 384 } = {}) {
  const nSrc = Math.max(1, Math.round(duration * srcFps));
  const n = Math.min(nSrc, Math.max(minFrames, Math.min(maxFrames, Math.floor(duration * fps))));
  const step = n > 1 ? (nSrc - 1) / (n - 1) : 0;
  const times = [];
  for (let i = 0; i < n; i++) times.push(roundEven(i * step) / srcFps);
  return { times, nSrc, srcFps };
}

/** printf("%.1f") of a time: ties of the exact value go to even, as in C. */
export function stamp(seconds) {
  const y = seconds * 10;
  let r = Math.round(y);
  if (Math.abs(y - Math.trunc(y)) === 0.5) r = 2 * Math.round(y / 2);
  return `<${(r / 10).toFixed(1)} seconds>`;
}

const roundEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };

/**
 * The groups of a clip: each has its timestamp text, its two frame indices (a short last group repeats its frame)
 * and its slice [start, end) of the soundtrack in samples at `rate`.
 * @param {number[]} times frame times from frameTimes
 * @param {number} audioLength samples in the soundtrack (0 for none)
 */
export function groups(times, audioLength = 0, rate = 16000) {
  const out = [];
  const nG = Math.ceil(times.length / 2);
  const start = (g) => Math.min(audioLength, Math.round(times[g * 2] * rate));
  for (let g = 0; g < nG; g++) {
    const a = g * 2, b = Math.min(a + 1, times.length - 1);
    out.push({
      text: stamp((times[a] + times[b]) / 2),
      frames: [a, b],
      audio: [g === 0 ? 0 : start(g), g === nG - 1 ? audioLength : start(g + 1)],
    });
  }
  return out;
}

/**
 * Reads a video in the browser: frames at the sampled times, scaled to fit `maxPixels` (multiples of 32), and the
 * soundtrack at 16 kHz mono if it has one.
 * @param {Blob | string} source a file/blob or a URL
 * @param {{ maxPixels?: number, minPixels?: number, fps?: number, srcFps?: number, maxFrames?: number, audio?: boolean,
 *   onStatus?: (s: string) => void }} [o]
 * @returns {Promise<{ frames: ImageData[], times: number[], audio: Float32Array | null, duration: number,
 *   width: number, height: number }>}
 */
export async function readVideo(source, { maxPixels = 256 * 256, minPixels = 4096, fps = 2, srcFps = 30, maxFrames = 384, audio = true, onStatus } = {}) {
  const url = typeof source === 'string' ? source : URL.createObjectURL(source);
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto'; v.crossOrigin = 'anonymous';
  try {
    v.src = url;
    await once(v, 'loadeddata');
    // iOS Safari can hand canvas a blank frame from a video that has never played: start it (muted) and stop.
    try { await v.play(); } catch { /* not allowed or not needed */ }
    v.pause();
    let duration = v.duration;
    if (!Number.isFinite(duration)) { // some WebM files only learn their length by seeking to the end
      v.currentTime = 1e9; await once(v, 'seeked'); duration = v.duration; v.currentTime = 0; await once(v, 'seeked');
    }
    const { times } = frameTimes(duration, { fps, srcFps, maxFrames });
    const size = smartResize(v.videoWidth, v.videoHeight, 32, minPixels, maxPixels);
    const canvas = new OffscreenCanvas(size.width, size.height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    const frames = [];
    for (let i = 0; i < times.length; i++) {
      onStatus?.(`reading frame ${i + 1} of ${times.length}`);
      // Aim inside the frame (half a source frame in), so the decoder shows that frame and not the one before.
      v.currentTime = Math.min(duration, times[i] + 0.5 / srcFps);
      await once(v, 'seeked');
      ctx.drawImage(v, 0, 0, size.width, size.height);
      frames.push(ctx.getImageData(0, 0, size.width, size.height));
    }
    let pcm = null;
    if (audio) {
      try {
        const blob = typeof source === 'string' ? await (await fetch(source)).blob() : source;
        pcm = await decodeAudio(blob);
        // Only the span of the video stream counts (llama.cpp set_soundtrack).
        const len = Math.round(Math.round(duration * srcFps) / srcFps * 16000);
        if (pcm.length > len) pcm = pcm.subarray(0, len);
        if (!pcm.length) pcm = null;
      } catch { pcm = null; } // no audio track, or a codec the browser can't decode
    }
    return { frames, times, audio: pcm, duration, width: size.width, height: size.height };
  } finally {
    v.removeAttribute('src'); v.load();
    if (typeof source !== 'string') URL.revokeObjectURL(url);
  }
}

function once(el, ev) {
  return new Promise((res, rej) => {
    const ok = () => { el.removeEventListener('error', bad); res(); };
    const bad = () => { el.removeEventListener(ev, ok); rej(new Error(`video: ${el.error?.message || 'could not be read'}`)); };
    el.addEventListener(ev, ok, { once: true });
    el.addEventListener('error', bad, { once: true });
  });
}
