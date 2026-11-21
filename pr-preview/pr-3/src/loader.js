// Streams GGUF tensors into GPU buffers, one piece at a time, through a moving read window.

import { repackQ4_0, repackQ6_K, repackQ8_0, f16BytesToF32 } from './quant.js';

const WINDOW_READ = 16 << 20; // bytes per HTTP/file read
const PIECE = 8 << 20; // max bytes repacked at once

/** Reads a source through one moving window so neighbouring small tensors share a request. */
export function windowed(src, fileEnd) {
  let start = 0, buf = new Uint8Array(0);
  return async (offset, length) => {
    if (offset >= start && offset + length <= start + buf.length) return buf.subarray(offset - start, offset - start + length);
    buf = null; // drop the old window before reading the next one
    const n = Math.min(Math.max(length, WINDOW_READ), fileEnd - offset);
    buf = await src.read(offset, n);
    start = offset;
    return buf.subarray(0, length);
  };
}

/**
 * Loads `tensors` (GGUF tensor infos) onto the GPU. Returns a map name → { buf, type, rows, K, dOff }.
 * Quantized tensors keep their quant format (repacked); F16 becomes F32.
 * @param {import('./gpu.js').Gpu} gpu
 */
export async function streamTensors(gpu, src, tensors, { tag = 'weights', onProgress } = {}) {
  const q = gpu.device.queue;
  const list = [...tensors].sort((a, b) => a.offset - b.offset);
  const end = list.at(-1).offset + list.at(-1).nbytes;
  const total = list.reduce((s, t) => s + t.nbytes, 0);
  const read = windowed(src, end);
  const out = {};
  let loaded = 0;
  for (const t of list) {
    const rows = t.dims.length > 1 ? t.count / t.dims[0] : 1;
    const K = t.dims[0];
    let rec;
    if (t.typeName === 'Q4_0' || t.typeName === 'Q8_0') {
      const q8 = t.typeName === 'Q8_0';
      const bb = q8 ? 34 : 18, qb = q8 ? 32 : 16, blocks = t.count / 32;
      const buf = gpu.buffer(blocks * qb + Math.ceil(blocks / 2) * 4, tag);
      rec = { buf, type: t.typeName, dOff: (blocks * qb) / 4, rows, K };
      const step = Math.floor(PIECE / (bb * 2)) * 2; // even block count per piece
      for (let b0 = 0; b0 < blocks; b0 += step) {
        const n = Math.min(step, blocks - b0);
        const { d, qs } = (q8 ? repackQ8_0 : repackQ4_0)(await read(t.offset + b0 * bb, n * bb));
        q.writeBuffer(buf, b0 * qb, qs);
        q.writeBuffer(buf, blocks * qb + b0 * 2, d);
      }
    } else if (t.typeName === 'Q6_K') {
      const blocks = t.count / 256;
      const buf = gpu.buffer(blocks * 212, tag);
      rec = { buf, type: 'Q6_K', rows, K };
      const step = Math.floor(PIECE / 210);
      for (let b0 = 0; b0 < blocks; b0 += step) {
        const n = Math.min(step, blocks - b0);
        q.writeBuffer(buf, b0 * 212, repackQ6_K(await read(t.offset + b0 * 210, n * 210)).words);
      }
    } else if (t.typeName === 'F32' || t.typeName === 'F16') {
      const buf = gpu.buffer(t.count * 4, tag);
      rec = { buf, type: 'F32', rows, K };
      const bytes = await read(t.offset, t.nbytes);
      q.writeBuffer(buf, 0, t.typeName === 'F16' ? f16BytesToF32(bytes) : bytes.slice());
    } else {
      throw new Error(`${t.name}: ${t.typeName} weights are not supported yet`);
    }
    out[t.name] = rec;
    loaded += t.nbytes;
    // Let the GPU take the writes before reading more, so staging copies don't pile up.
    if (loaded % (64 << 20) < t.nbytes) await gpu.done();
    onProgress?.({ loaded, total, tensor: t.name });
  }
  await gpu.done();
  return out;
}

/** Frees every buffer in a map returned by streamTensors. */
export function freeTensors(gpu, w) {
  for (const r of Object.values(w)) gpu.free(r.buf);
}
