// CPU kernels for the WebAssembly backend (src/cpu.js): the matmuls, which are nearly all of a forward pass.
// Like llama.cpp's CPU path, activations are quantized to int8 first (Q8_0 blocks of 32 for Q4_0 weights, Q8_K
// blocks of 256 for Q6_K) so the inner loops are int8 dot products in 128-bit SIMD. Weights are read in the GPU
// layouts the loader already writes (src/quant.js), so CPU and GPU share one loader.
//
// Build (clang 16+ with the wasm32 target; the output is committed, so nobody else needs a toolchain):
//   clang --target=wasm32 -O3 -msimd128 -nostdlib -fno-builtin -Wl,--no-entry -Wl,--import-memory \
//     -Wl,--export-dynamic -Wl,--export=__heap_base -Wl,-z,stack-size=65536 -Wl,--strip-all -o src/cpu.wasm src/cpu.c
// The C stack sits at the bottom of the memory; src/cpu.js hands out buffers only above __heap_base.
#include <stdint.h>
#include <wasm_simd128.h>

#define EXPORT __attribute__((visibility("default")))

static float f16slow(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  union { uint32_t u; float f; } v;
  if (e == 0) { v.f = (float)m * (1.0f / 16777216.0f); v.u |= s; return v.f; } // subnormal or zero
  v.u = e == 31 ? (s | 0x7f800000 | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
  return v.f;
}

// f16 → f32 for quant scales: normal numbers by moving bits, the rare zero/subnormal/inf the slow way.
static inline float f16(uint16_t h) {
  const uint32_t e = h & 0x7c00;
  if (__builtin_expect(e != 0 && e != 0x7c00, 1)) {
    union { uint32_t u; float f; } v = { ((uint32_t)(h & 0x8000) << 16) | (((uint32_t)(h & 0x7fff) << 13) + (112u << 23)) };
    return v.f;
  }
  return f16slow(h);
}

// float → f16 → float, round to nearest even (llama.cpp keeps Q8_0 scales as f16).
static inline float roundF16(float f) {
  union { float f; uint32_t u; } v = { f };
  uint32_t x = v.u, s = x & 0x80000000;
  int32_t e = (int32_t)((x >> 23) & 0xff) - 127;
  if (e < -14 || e > 15) return f; // scales here never get near the f16 limits
  uint32_t lo = x & 0x1fff, keep = x & ~0x1fffu;
  if (lo > 0x1000 || (lo == 0x1000 && (x & 0x2000))) keep += 0x2000;
  v.u = keep | s;
  return v.f;
}

static inline float fabsf_(float x) { return x < 0 ? -x : x; }
static inline int nearest(float x) { return (int)(x + (x >= 0 ? 0.5f : -0.5f)); }

/** Q8_0: T rows of K floats (row stride xStride) → int8 xq[T][K], one scale per 32 in xd[T][K/32]. */
EXPORT void quant_q8_0(const float *x, int xStride, int K, int T, int8_t *xq, float *xd) {
  for (int t = 0; t < T; t++) {
    const float *r = x + (long)t * xStride;
    for (int b = 0; b < K / 32; b++) {
      float amax = 0;
      for (int j = 0; j < 32; j++) { float a = fabsf_(r[b * 32 + j]); if (a > amax) amax = a; }
      float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
      for (int j = 0; j < 32; j++) xq[(long)t * K + b * 32 + j] = (int8_t)nearest(r[b * 32 + j] * id);
      xd[(long)t * (K / 32) + b] = roundF16(d);
    }
  }
}

/** Q8_K: one row of K floats → int8, one f32 scale per 256 (ggml's quantize_row_q8_K). */
EXPORT void quant_q8_k(const float *x, int K, int8_t *xq, float *xd) {
  for (int b = 0; b < K / 256; b++) {
    const float *r = x + b * 256;
    float amax = 0, max = 0;
    for (int j = 0; j < 256; j++) { float a = fabsf_(r[j]); if (a > amax) { amax = a; max = r[j]; } }
    if (!amax) { for (int j = 0; j < 256; j++) xq[b * 256 + j] = 0; xd[b] = 0; continue; }
    float is = -127.0f / max;
    for (int j = 0; j < 256; j++) { int v = nearest(is * r[j]); xq[b * 256 + j] = (int8_t)(v > 127 ? 127 : v); }
    xd[b] = 1.0f / is;
  }
}

// Σ w·x for one Q4_0 block (lo = values 0..15, hi = 16..31, both in [-8, 7]) against 32 int8 x, as 4 int32 lanes.
// The four products per i16 lane add up to at most 4·8·127 = 4064, so they are summed in i16 before widening.
static inline v128_t dotq4(v128_t lo, v128_t hi, const int8_t *x) {
  const v128_t x0 = wasm_v128_load(x), x1 = wasm_v128_load(x + 16);
  const v128_t s = wasm_i16x8_add(wasm_i16x8_add(wasm_i16x8_extmul_low_i8x16(lo, x0), wasm_i16x8_extmul_high_i8x16(lo, x0)),
                                  wasm_i16x8_add(wasm_i16x8_extmul_low_i8x16(hi, x1), wasm_i16x8_extmul_high_i8x16(hi, x1)));
  return wasm_i32x4_extadd_pairwise_i16x8(s);
}

// Σ a·b over 16 int8 lanes, as 4 int32 lanes.
static inline v128_t dot16(v128_t a, v128_t b) {
  return wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_add(wasm_i16x8_extmul_low_i8x16(a, b), wasm_i16x8_extmul_high_i8x16(a, b)));
}
static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

// Rows [ra, rb) × tokens [ta, tb), one row and up to 4 tokens at a time: the general case and leftovers.
static void q4rows(const uint8_t *qs, const uint16_t *d, int ra, int rb, int ta, int tb, int K,
                   const int8_t *xq, const float *xd, float *Y, int yStride, int yOff, int accumulate) {
  const int nb = K / 32;
  const v128_t m4 = wasm_i8x16_splat(15), eight = wasm_i8x16_splat(8);
  for (int r = ra; r < rb; r++) {
    const uint8_t *wq = qs + (long)r * nb * 16;
    const uint16_t *wd = d + (long)r * nb;
    for (int t0 = ta; t0 < tb; t0 += 4) {
      const int nt = tb - t0 < 4 ? tb - t0 : 4;
      v128_t acc[4] = { wasm_f32x4_splat(0), wasm_f32x4_splat(0), wasm_f32x4_splat(0), wasm_f32x4_splat(0) };
      for (int b = 0; b < nb; b++) {
        const v128_t q = wasm_v128_load(wq + b * 16);
        const v128_t lo = wasm_i8x16_sub(wasm_v128_and(q, m4), eight);
        const v128_t hi = wasm_i8x16_sub(wasm_u8x16_shr(q, 4), eight);
        const float dw = f16(wd[b]);
        for (int t = 0; t < nt; t++) {
          const v128_t s = dotq4(lo, hi, xq + (long)(t0 + t) * K + b * 32);
          acc[t] = wasm_f32x4_add(acc[t], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(s), wasm_f32x4_splat(dw * xd[(long)(t0 + t) * nb + b])));
        }
      }
      for (int t = 0; t < nt; t++) {
        float *y = Y + (long)(t0 + t) * yStride + yOff + r;
        *y = accumulate ? *y + hsum(acc[t]) : hsum(acc[t]);
      }
    }
  }
}

/**
 * Y[t·yStride + yOff + r] (=, or += with accumulate) Σ_k W[r, k]·X[t, k] for r in [r0, r1), with X already
 * quantized by quant_q8_0. W is Q4_0 in the GPU layout: 16 bytes of nibbles per block at `qs` (byte j = value j
 * low nibble, value j+16 high nibble), f16 scales at `d`. Decode does 4 rows per pass so they share each block of
 * x (2× faster than one row); prompts do one row × 4 tokens per pass (2 rows × 4 tokens spilled registers on x86
 * and was slower).
 */
EXPORT void mm_q4_0(const uint8_t *qs, const uint16_t *d, int r0, int r1, int K, int T,
                    const int8_t *xq, const float *xd, float *Y, int yStride, int yOff, int accumulate) {
  const int nb = K / 32;
  const v128_t m4 = wasm_i8x16_splat(15), eight = wasm_i8x16_splat(8);
#define UNPACK(lo, hi, p) const v128_t q_##lo = wasm_v128_load(p); \
  const v128_t lo = wasm_i8x16_sub(wasm_v128_and(q_##lo, m4), eight), hi = wasm_i8x16_sub(wasm_u8x16_shr(q_##lo, 4), eight);
#define FMA(acc, s, scale) acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(s), wasm_f32x4_splat(scale)))
  if (T == 1) {
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
      v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
      const uint8_t *w = qs + (long)r * nb * 16;
      const uint16_t *dd = d + (long)r * nb;
      for (int b = 0; b < nb; b++) {
        const int8_t *x = xq + b * 32;
        const float dx = xd[b];
        UNPACK(l0, h0, w + (long)b * 16) UNPACK(l1, h1, w + ((long)nb + b) * 16)
        UNPACK(l2, h2, w + ((long)2 * nb + b) * 16) UNPACK(l3, h3, w + ((long)3 * nb + b) * 16)
        FMA(a0, dotq4(l0, h0, x), f16(dd[b]) * dx);
        FMA(a1, dotq4(l1, h1, x), f16(dd[nb + b]) * dx);
        FMA(a2, dotq4(l2, h2, x), f16(dd[2 * nb + b]) * dx);
        FMA(a3, dotq4(l3, h3, x), f16(dd[3 * nb + b]) * dx);
      }
      float *y = Y + yOff + r;
      if (accumulate) { y[0] += hsum(a0); y[1] += hsum(a1); y[2] += hsum(a2); y[3] += hsum(a3); }
      else { y[0] = hsum(a0); y[1] = hsum(a1); y[2] = hsum(a2); y[3] = hsum(a3); }
    }
    q4rows(qs, d, r, r1, 0, 1, K, xq, xd, Y, yStride, yOff, accumulate);
    return;
  }
#undef UNPACK
#undef FMA
  q4rows(qs, d, r0, r1, 0, T, K, xq, xd, Y, yStride, yOff, accumulate);
}

/**
 * Y[r] = Σ_k W[r, k]·x[k] for r in [r0, r1), W a padded Q6_K matrix (212-byte blocks: ql[128], qh[64],
 * scales[16], d at 208) and x quantized by quant_q8_k. Each 16-value group of a block has one int8 scale.
 */
EXPORT void mv_q6_k(const uint8_t *W, int r0, int r1, int K, const int8_t *xq, const float *xd, float *Y) {
  const int nb = K / 256;
  const v128_t m4 = wasm_i8x16_splat(15), m3 = wasm_i8x16_splat(3), off = wasm_i8x16_splat(32);
  for (int r = r0; r < r1; r++) {
    float acc = 0;
    for (int b = 0; b < nb; b++) {
      const uint8_t *blk = W + ((long)r * nb + b) * 212;
      const int8_t *x = xq + b * 256;
      v128_t isum = wasm_i32x4_splat(0);
      for (int half = 0; half < 2; half++) {
        const uint8_t *ql = blk + half * 64, *qh = blk + 128 + half * 32;
        const int8_t *sc = (const int8_t *)(blk + 192 + half * 8);
        const int8_t *xh = x + half * 128;
        for (int g = 0; g < 2; g++) {
          const v128_t l0 = wasm_v128_load(ql + g * 16), l1 = wasm_v128_load(ql + 32 + g * 16);
          const v128_t h = wasm_v128_load(qh + g * 16);
          const v128_t q1 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(l0, m4), wasm_i8x16_shl(wasm_v128_and(h, m3), 4)), off);
          const v128_t q2 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(l1, m4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(h, 2), m3), 4)), off);
          const v128_t q3 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(l0, 4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(h, 4), m3), 4)), off);
          const v128_t q4 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(l1, 4), wasm_i8x16_shl(wasm_u8x16_shr(h, 6), 4)), off);
          const int8_t *xg = xh + g * 16;
          isum = wasm_i32x4_add(isum, wasm_i32x4_mul(dot16(q1, wasm_v128_load(xg)), wasm_i32x4_splat(sc[g])));
          isum = wasm_i32x4_add(isum, wasm_i32x4_mul(dot16(q2, wasm_v128_load(xg + 32)), wasm_i32x4_splat(sc[g + 2])));
          isum = wasm_i32x4_add(isum, wasm_i32x4_mul(dot16(q3, wasm_v128_load(xg + 64)), wasm_i32x4_splat(sc[g + 4])));
          isum = wasm_i32x4_add(isum, wasm_i32x4_mul(dot16(q4, wasm_v128_load(xg + 96)), wasm_i32x4_splat(sc[g + 6])));
        }
      }
      const int s = wasm_i32x4_extract_lane(isum, 0) + wasm_i32x4_extract_lane(isum, 1) + wasm_i32x4_extract_lane(isum, 2) + wasm_i32x4_extract_lane(isum, 3);
      acc += f16(blk[208] | (blk[209] << 8)) * xd[b] * (float)s;
    }
    Y[r] = acc;
  }
}
