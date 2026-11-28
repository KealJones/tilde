// Repacking GGUF quant blocks into GPU-friendly, 4-byte-aligned layouts, and CPU reference dequant for tests.
//
// GPU layouts (all little-endian u32 words):
//   Q4_0 → two arrays: `qs` (4 words per 32-value block: byte j = value j low nibble, value j+16 high nibble)
//          and `d` (one f16 scale per block, two per word; odd counts padded).
//   Q6_K → one array, each 210-byte block padded to 212 bytes (53 words); fields stay where ggml put them:
//          ql[128] at 0, qh[64] at 128, scales[16] (int8) at 192, d (f16) at 208.

export const Q4_0 = { block: 32, bytes: 18 };
export const Q6_K = { block: 256, bytes: 210, padded: 212 };

/** Splits Q4_0 blocks into scale and quant arrays. Works on any whole number of blocks. */
export function repackQ4_0(src) {
  const n = src.length / Q4_0.bytes;
  if (!Number.isInteger(n)) throw new Error('Q4_0 data is not a whole number of blocks');
  const d = new Uint16Array(n + (n & 1));
  const qs = new Uint8Array(n * 16);
  for (let b = 0, o = 0; b < n; b++, o += 18) {
    d[b] = src[o] | (src[o + 1] << 8);
    qs.set(src.subarray(o + 2, o + 18), b * 16);
  }
  return { d: new Uint32Array(d.buffer), qs: new Uint32Array(qs.buffer), blocks: n };
}

/** Pads each Q6_K block to 212 bytes. */
export function repackQ6_K(src) {
  const n = src.length / Q6_K.bytes;
  if (!Number.isInteger(n)) throw new Error('Q6_K data is not a whole number of blocks');
  const out = new Uint8Array(n * Q6_K.padded);
  for (let b = 0; b < n; b++) out.set(src.subarray(b * 210, b * 210 + 210), b * 212);
  return { words: new Uint32Array(out.buffer), blocks: n };
}

const F16 = new Float32Array(1);
const F16I = new Uint32Array(F16.buffer);
/** IEEE half → float. */
export function f16ToF32(h) {
  const s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) { if (m === 0) { F16I[0] = s; return F16[0]; } return (s ? -1 : 1) * m * 2 ** -24; }
  if (e === 31) { F16I[0] = s | 0x7f800000 | (m << 13); return F16[0]; }
  F16I[0] = s | ((e + 112) << 23) | (m << 13);
  return F16[0];
}

/** float → IEEE half (round to nearest even). */
export function f32ToF16(f) {
  F16[0] = f;
  const x = F16I[0];
  const s = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e >= 31) return s | 0x7c00 | (((x >>> 23) & 0xff) === 255 && m ? 0x200 : 0);
  if (e <= 0) {
    if (e < -10) return s;
    m |= 0x800000;
    const shift = 14 - e;
    let r = m >> shift;
    const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (r & 1))) r++;
    return s | r;
  }
  let r = (e << 10) | (m >> 13);
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (r & 1))) r++;
  return s | r;
}

/** Reference: Q4_0 blocks → float32. */
export function dequantQ4_0(src) {
  const n = src.length / 18;
  const out = new Float32Array(n * 32);
  for (let b = 0; b < n; b++) {
    const o = b * 18;
    const d = f16ToF32(src[o] | (src[o + 1] << 8));
    for (let j = 0; j < 16; j++) {
      const q = src[o + 2 + j];
      out[b * 32 + j] = ((q & 0xf) - 8) * d;
      out[b * 32 + j + 16] = ((q >> 4) - 8) * d;
    }
  }
  return out;
}

/** Reference: Q6_K blocks → float32 (ggml's dequantize_row_q6_K). */
export function dequantQ6_K(src) {
  const n = src.length / 210;
  const out = new Float32Array(n * 256);
  for (let b = 0; b < n; b++) {
    const o = b * 210;
    const d = f16ToF32(src[o + 208] | (src[o + 209] << 8));
    for (let half = 0; half < 2; half++) {
      const ql = o + half * 64, qh = o + 128 + half * 32, sc = o + 192 + half * 8;
      const y = b * 256 + half * 128;
      for (let l = 0; l < 32; l++) {
        const is = l >> 4;
        const s = (i) => (src[sc + i] << 24) >> 24;
        const q1 = ((src[ql + l] & 0xf) | (((src[qh + l] >> 0) & 3) << 4)) - 32;
        const q2 = ((src[ql + l + 32] & 0xf) | (((src[qh + l] >> 2) & 3) << 4)) - 32;
        const q3 = ((src[ql + l] >> 4) | (((src[qh + l] >> 4) & 3) << 4)) - 32;
        const q4 = ((src[ql + l + 32] >> 4) | (((src[qh + l] >> 6) & 3) << 4)) - 32;
        out[y + l] = d * s(is) * q1;
        out[y + l + 32] = d * s(is + 2) * q2;
        out[y + l + 64] = d * s(is + 4) * q3;
        out[y + l + 96] = d * s(is + 6) * q4;
      }
    }
  }
  return out;
}

export const Q8_0 = { block: 32, bytes: 34 };

/** Splits Q8_0 blocks into int8 quant words (8 per block) and f16 scale pairs. */
export function repackQ8_0(src) {
  const n = src.length / Q8_0.bytes;
  if (!Number.isInteger(n)) throw new Error('Q8_0 data is not a whole number of blocks');
  const d = new Uint16Array(n + (n & 1));
  const qs = new Uint8Array(n * 32);
  for (let b = 0, o = 0; b < n; b++, o += 34) {
    d[b] = src[o] | (src[o + 1] << 8);
    qs.set(src.subarray(o + 2, o + 34), b * 32);
  }
  return { d: new Uint32Array(d.buffer), qs: new Uint32Array(qs.buffer), blocks: n };
}

/** Reference: Q8_0 blocks → float32. */
export function dequantQ8_0(src) {
  const n = src.length / 34;
  const out = new Float32Array(n * 32);
  for (let b = 0; b < n; b++) {
    const d = f16ToF32(src[b * 34] | (src[b * 34 + 1] << 8));
    for (let j = 0; j < 32; j++) out[b * 32 + j] = ((src[b * 34 + 2 + j] << 24) >> 24) * d;
  }
  return out;
}

/** F16 bytes → Float32Array. */
export function f16BytesToF32(src) {
  const h = new Uint16Array(src.buffer, src.byteOffset, src.length / 2);
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = f16ToF32(h[i]);
  return out;
}
