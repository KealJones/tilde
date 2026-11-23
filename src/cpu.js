// CPU device: the same surface as Gpu (src/gpu.js), backed by one WebAssembly memory, so the model files run on
// it unchanged. Used when the browser has no WebGPU (Firefox on Android, a GPU on the blocklist), or on request.
//
// Every "buffer" is a counted range of that memory. A dispatch looks its WGSL source up in kernelInfo
// (src/kernels.js) and runs this file's version of that kernel, at the time the GPU would: when the command
// encoder is run. The matmuls, nearly all of the work, are WASM SIMD (src/cpu.c); the rest is plain JS.

import * as K from './kernels.js';
import { f16ToF32, f32ToF16 } from './quant.js';

const PAGE = 65536;
const ALIGN = 64;
const GROW = 16 << 20; // grow the memory in steps of at least this much

let wasmBytes;
async function loadWasm() {
  const url = new URL('./cpu.wasm', import.meta.url);
  if (url.protocol === 'file:') return (await import('node:fs/promises')).readFile(url); // node tests
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

/** True when this JS engine runs WebAssembly with 128-bit SIMD (every current browser; Safari since 16.4). */
export function cpuSupported() {
  try {
    // A function returning a v128: validates only with SIMD.
    return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 22, 1, 20, 0, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11]));
  } catch { return false; }
}

/** Opens the CPU device. */
export async function openCpu() {
  if (typeof WebAssembly !== 'object' || !cpuSupported()) throw new Error('This browser has neither WebGPU nor WebAssembly SIMD, so Tilde cannot run here.');
  wasmBytes ||= await loadWasm();
  const memory = new WebAssembly.Memory({ initial: 256 });
  const { instance } = await WebAssembly.instantiate(wasmBytes, { env: { memory } });
  return new Cpu(memory, instance.exports);
}

/** Records copies and dispatches; they run when the encoder is submitted or passed to Cpu#run. */
class Encoder {
  constructor() { this.ops = []; }
  copyBufferToBuffer(src, srcOff, dst, dstOff, n) {
    this.ops.push((cpu) => cpu.u8().copyWithin(dst.off + dstOff, src.off + srcOff, src.off + srcOff + n));
  }
  finish() { return this; }
}

export class Cpu {
  constructor(memory, wasm) {
    this.backend = 'cpu';
    this.memory = memory;
    this.wasm = wasm;
    this.bytes = new Map();
    this.total = 0;
    this.peak = 0;
    this.longestSubmit = 0;
    this.lost = null;
    // The C stack and statics live below __heap_base (src/cpu.c); buffers start above it.
    this.top = Math.ceil(wasm.__heap_base.value / ALIGN) * ALIGN;
    this.holes = []; // freed ranges [{ off, size }], sorted by offset
    this.views = null;
    this.scratch = null;
    const cpu = this;
    // The parts of a GPUDevice the engine uses.
    this.device = {
      limits: { maxBufferSize: 2 ** 32, maxStorageBufferBindingSize: 2 ** 32 },
      createCommandEncoder: () => new Encoder(),
      queue: {
        writeBuffer(buf, off, data, dataOff = 0, size) {
          const bytes = ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset + dataOff * data.BYTES_PER_ELEMENT, size === undefined ? data.byteLength - dataOff * data.BYTES_PER_ELEMENT : size * data.BYTES_PER_ELEMENT)
            : new Uint8Array(data, dataOff, size ?? data.byteLength - dataOff);
          if (off + bytes.length > buf.size) throw new Error('writeBuffer past the end of the buffer');
          cpu.u8().set(bytes, buf.off + off);
        },
        submit(encs) { for (const e of encs) for (const op of e.ops) op(cpu); },
        onSubmittedWorkDone: async () => {},
      },
    };
  }

  get maxBinding() { return this.device.limits.maxStorageBufferBindingSize; }

  /** Typed views of the whole memory (made again after the memory grows). */
  #views() {
    const b = this.memory.buffer;
    if (this.views?.b !== b) this.views = { b, u8: new Uint8Array(b), u32: new Uint32Array(b), f32: new Float32Array(b) };
    return this.views;
  }
  u8() { return this.#views().u8; }
  u32() { return this.#views().u32; }
  f32() { return this.#views().f32; }

  /** Allocates a buffer in the memory and counts it under `tag`. */
  buffer(size, tag = 'misc') {
    const padded = Math.max(16, Math.ceil(size / 16) * 16);
    const n = Math.ceil(padded / ALIGN) * ALIGN;
    let off = -1;
    const i = this.holes.findIndex((h) => h.size >= n);
    if (i >= 0) {
      const h = this.holes[i];
      off = h.off;
      if (h.size === n) this.holes.splice(i, 1); else { h.off += n; h.size -= n; }
    } else {
      off = this.top;
      if (off + n > 2 ** 32) throw new Error(`${tag}: out of WebAssembly memory (4 GB)`);
      const need = off + n - this.memory.buffer.byteLength;
      if (need > 0) {
        try { this.memory.grow(Math.ceil(Math.max(need, GROW) / PAGE)); } catch {
          try { this.memory.grow(Math.ceil(need / PAGE)); } catch { throw new Error(`${tag}: the browser would not give the CPU backend ${Math.round((off + n) / 1048576)} MB`); }
        }
      }
      this.top = off + n;
    }
    this.u8().fill(0, off, off + n); // fresh GPU buffers are zeroed; match that
    const b = { off, size: padded, span: n, __tag: tag, __freed: false, destroy() {} };
    this.bytes.set(tag, (this.bytes.get(tag) || 0) + padded);
    this.total += padded;
    this.peak = Math.max(this.peak, this.total);
    return b;
  }

  /** Frees a buffer made by `buffer()`; its range is reused by later buffers. */
  free(b) {
    if (!b || b.__freed) return;
    b.__freed = true;
    this.bytes.set(b.__tag, this.bytes.get(b.__tag) - b.size);
    this.total -= b.size;
    const hs = this.holes;
    let i = hs.findIndex((h) => h.off > b.off);
    if (i < 0) i = hs.length;
    hs.splice(i, 0, { off: b.off, size: b.span });
    // Merge with the neighbours, and give the end back to the bump pointer.
    if (i + 1 < hs.length && hs[i].off + hs[i].size === hs[i + 1].off) { hs[i].size += hs[i + 1].size; hs.splice(i + 1, 1); }
    if (i > 0 && hs[i - 1].off + hs[i - 1].size === hs[i].off) { hs[i - 1].size += hs[i].size; hs.splice(i, 1); i--; }
    if (hs[i].off + hs[i].size === this.top) { this.top = hs[i].off; hs.splice(i, 1); }
  }

  /** A scratch range of at least n bytes, reused between kernels (not counted: it is CPU-only working space). */
  scratchAt(n) {
    if (!this.scratch || this.scratch.size < n) {
      if (this.scratch) this.free(this.scratch);
      this.scratch = this.buffer(Math.max(n, 1 << 20), 'cpu-scratch');
    }
    return this.scratch.off;
  }

  beginPlan() {}
  endPass() {}
  flush() {}
  done() { return Promise.resolve(); }

  /** Records one dispatch: the CPU version of the kernel whose WGSL source is `code`. */
  dispatch(enc, code, bindings, groups) {
    let info = K.kernelInfo.get(code);
    if (!info) {
      for (const [name, v] of Object.entries(K)) if (typeof v === 'string' && !K.kernelInfo.has(v)) K.kernelInfo.set(v, { name, args: [] });
      info = K.kernelInfo.get(code);
    }
    const run = info && KERNELS[info.name];
    if (!run) throw new Error(`The CPU backend has no "${info?.name || 'unknown'}" kernel yet; this needs WebGPU.`);
    const [u, ...bufs] = bindings;
    const uw = Uint32Array.from(new Uint32Array(u.buffer, u.byteOffset, u.length));
    const g = Array.isArray(groups) ? groups : [groups];
    enc.ops.push((cpu) => run(cpu, uw, new Float32Array(uw.buffer), bufs, g, info.args));
  }

  /** Runs `record(encoder)`, yielding to the page every ~50 ms so it stays responsive. Returns the ms it took. */
  async run(record, enc = new Encoder()) {
    record(enc);
    const t0 = performance.now();
    let last = t0;
    for (const op of enc.ops) {
      op(this);
      const now = performance.now();
      if (now - last > 50) { await new Promise((r) => setTimeout(r, 0)); last = performance.now(); }
    }
    const ms = performance.now() - t0;
    this.longestSubmit = Math.max(this.longestSubmit, ms);
    return ms;
  }

  /** Copies a buffer out. */
  async read(buf, size = buf.size, offset = 0) {
    return this.u8().slice(buf.off + offset, buf.off + offset + size).buffer;
  }

  /** Memory report in MB by tag, like Gpu#report, plus the WebAssembly memory's size. */
  report() {
    const mb = (n) => Math.round(n / 1048576);
    return { backend: 'cpu', totalMB: mb(this.total), peakMB: mb(this.peak), heapMB: mb(this.memory.buffer.byteLength), byTag: Object.fromEntries([...this.bytes].map(([k, v]) => [k, mb(v)])), longestSubmitMs: Math.round(this.longestSubmit) };
  }
}

// ───────────── Kernels: the same names, uniforms and bindings as the WGSL in src/kernels.js ─────────────

const H2F = new Float32Array(65536);
for (let h = 0; h < 65536; h++) H2F[h] = f16ToF32(h);
const fi = (b) => b.off >> 2; // float/word index of a buffer's start

// Q4_0 matmuls. matvecQ4_0 is single-token and writes Y[yOff + row]; the gemms overwrite; matmulQ4_0 can accumulate.
function q4(name) {
  return (cpu, u, f, [W, X, Y]) => {
    const [rows, Kd, T, dOff, yStride, yOff, xStride, accumulate] = u;
    const nb = Kd / 32, w = cpu.wasm;
    const s = cpu.scratchAt(T * Kd + T * nb * 4 + 64);
    const xq = s, xd = s + Math.ceil((T * Kd) / 16) * 16;
    w.quant_q8_0(X.off, xStride, Kd, T, xq, xd);
    w.mm_q4_0(W.off, W.off + dOff * 4, 0, rows, Kd, T, xq, xd, Y.off, yStride, yOff, name === 'matmulQ4_0' ? accumulate : 0);
  };
}

function dequantQ6KRow(cpu, W, row, Kd, out, o) {
  const u8 = cpu.u8(), nb = Kd / 256;
  for (let b = 0; b < nb; b++) {
    const base = W.off + (row * nb + b) * 212;
    const d = H2F[u8[base + 208] | (u8[base + 209] << 8)];
    for (let half = 0; half < 2; half++) {
      const ql = base + half * 64, qh = base + 128 + half * 32, sc = base + 192 + half * 8;
      const y = o + b * 256 + half * 128;
      for (let l = 0; l < 32; l++) {
        const is = l >> 4, s = (i) => (u8[sc + i] << 24) >> 24, h = u8[qh + l];
        out[y + l] = d * s(is) * (((u8[ql + l] & 15) | ((h & 3) << 4)) - 32);
        out[y + l + 32] = d * s(is + 2) * (((u8[ql + l + 32] & 15) | (((h >> 2) & 3) << 4)) - 32);
        out[y + l + 64] = d * s(is + 4) * (((u8[ql + l] >> 4) | (((h >> 4) & 3) << 4)) - 32);
        out[y + l + 96] = d * s(is + 6) * (((u8[ql + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32);
      }
    }
  }
}

// Scaled dot-product attention for one query row against cache slots 0..L-1, with the same masks as the WGSL
// (causal, window, ring slots), written into O[ob..ob+127]: softmax(q·k / √128) · V, times `gate`.
function attendRow(cpu, Q, qb, KC, VC, kvBase, KVH, L, slots, maxPos, qp, window, O, ob, gate, scratch) {
  const f = cpu.f32(), u = cpu.u32();
  const kc = fi(KC), vc = fi(VC), stride = KVH * 64;
  let m = -Infinity;
  for (let j = 0; j < L; j++) {
    const kp = maxPos - ((maxPos + slots - (j % slots)) % slots);
    if (kp < 0 || kp > qp || qp - kp >= window) { scratch[j] = -Infinity; continue; }
    const kb = kc + j * stride + kvBase / 2;
    let acc = 0;
    for (let d = 0; d < 64; d++) {
      const w = u[kb + d];
      acc += f[qb + 2 * d] * H2F[w & 0xffff] + f[qb + 2 * d + 1] * H2F[w >>> 16];
    }
    scratch[j] = acc * 0.08838834764831845;
    if (scratch[j] > m) m = scratch[j];
  }
  let sum = 0;
  for (let j = 0; j < L; j++) { const e = scratch[j] === -Infinity ? 0 : Math.exp(scratch[j] - m); scratch[j] = e; sum += e; }
  const acc = new Float64Array(128);
  for (let j = 0; j < L; j++) {
    const p = scratch[j];
    if (!p) continue;
    const vb = vc + j * stride + kvBase / 2;
    for (let d = 0; d < 64; d++) { const w = u[vb + d]; acc[2 * d] += p * H2F[w & 0xffff]; acc[2 * d + 1] += p * H2F[w >>> 16]; }
  }
  const g = gate / sum;
  for (let d = 0; d < 128; d++) f[ob + d] = acc[d] * g;
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
let attnScratch = new Float32Array(4096);
const scores = (L) => (attnScratch.length >= L ? attnScratch : (attnScratch = new Float32Array(L * 2)));

const KERNELS = {
  matmulQ4_0: q4('matmulQ4_0'),
  gemmQ4_0: q4('gemmQ4_0'),
  gemmQ4_0x64: q4('gemmQ4_0x64'),
  matvecQ4_0: q4('matvecQ4_0'),

  matmulQ6_K(cpu, u, f, [W, X, Y]) {
    const [rows, Kd, xOff] = u;
    const s = cpu.scratchAt(Kd + (Kd / 256) * 4 + 64);
    cpu.wasm.quant_q8_k(X.off + xOff * 4, Kd, s, s + Kd);
    cpu.wasm.mv_q6_k(W.off, 0, rows, Kd, s, s + Kd, Y.off);
  },

  embedQ6_K(cpu, u, f, [W, tok, X]) {
    const [Kd, T] = u;
    for (let t = 0; t < T; t++) dequantQ6KRow(cpu, W, cpu.u32()[fi(tok) + t], Kd, cpu.f32(), fi(X) + t * Kd);
  },

  embedQ4_0(cpu, u, f, [W, tok, X], [T]) {
    const [Kd, dOff] = u, nb = Kd / 32, u8 = cpu.u8(), out = cpu.f32(), u16 = new Uint16Array(u8.buffer);
    for (let t = 0; t < T; t++) {
      const row = cpu.u32()[fi(tok) + t];
      for (let b = 0; b < nb; b++) {
        const blk = row * nb + b, d = H2F[u16[(W.off + dOff * 4) / 2 + blk]], q = W.off + blk * 16, o = fi(X) + t * Kd + b * 32;
        for (let j = 0; j < 16; j++) { out[o + j] = ((u8[q + j] & 15) - 8) * d; out[o + j + 16] = ((u8[q + j] >> 4) - 8) * d; }
      }
    }
  },

  rmsnorm(cpu, u, f, [X, w, Y], [rows]) {
    const [N, mode, , wMod, xOff, yOff] = u, eps = f[2], a = cpu.f32(), x0 = fi(X), w0 = fi(w), y0 = fi(Y);
    for (let r = 0; r < rows; r++) {
      const xb = x0 + xOff + r * N, yb = y0 + yOff + r * N;
      let s = 0;
      for (let i = 0; i < N; i++) s += a[xb + i] * a[xb + i];
      const inv = 1 / Math.sqrt(s / N + eps);
      for (let i = 0; i < N; i++) {
        const v = a[xb + i] * inv * a[w0 + (i % wMod)];
        if (mode === 1) a[yb + i] += v; else a[yb + i] = v;
      }
    }
  },

  rope(cpu, u, f, [V, factors], [rows]) {
    const [H, pos0] = u, base = f[2], a = cpu.f32(), v0 = fi(V), f0 = fi(factors);
    const inv = new Float64Array(64);
    for (let i = 0; i < 64; i++) inv[i] = base ** (-2 * i / 128) / a[f0 + i];
    for (let r = 0; r < rows; r++) {
      const pos = pos0 + Math.floor(r / H), b = v0 + r * 128;
      for (let i = 0; i < 64; i++) {
        const th = pos * inv[i], c = Math.cos(th), s = Math.sin(th), x = a[b + i], y = a[b + i + 64];
        a[b + i] = x * c - y * s;
        a[b + i + 64] = x * s + y * c;
      }
    }
  },

  kvWriteFor(cpu, u, f, [Kb, Vb, KC, VC], [T], [KVH]) {
    const [pos0, slots] = u, a = cpu.f32(), w = cpu.u32(), n = KVH * 64;
    for (let t = 0; t < T; t++) {
      const slot = (pos0 + t) % slots;
      for (let i = 0; i < n; i++) {
        const s = t * KVH * 128 + 2 * i;
        w[fi(KC) + slot * n + i] = f32ToF16(a[fi(Kb) + s]) | (f32ToF16(a[fi(Kb) + s + 1]) << 16);
        w[fi(VC) + slot * n + i] = f32ToF16(a[fi(Vb) + s]) | (f32ToF16(a[fi(Vb) + s + 1]) << 16);
      }
    }
  },

  attnScoresFor(cpu, u, f, [Q, KC, S], groups, [H, KVH]) {
    const [pos0, T, slots, L, window, sStride] = u, a = cpu.f32(), w = cpu.u32();
    const maxPos = pos0 + T - 1, stride = KVH * 64;
    for (let th = 0; th < T * H; th++) {
      const t = Math.floor(th / H), h = th % H, qp = pos0 + t, qb = fi(Q) + th * 128;
      for (let j = 0; j < L; j++) {
        const kp = maxPos - ((maxPos + slots - (j % slots)) % slots);
        let s = -3.0e38;
        if (kp >= 0 && kp <= qp && qp - kp < window) {
          const kb = fi(KC) + j * stride + Math.floor(h / (H / KVH)) * 64;
          let acc = 0;
          for (let d = 0; d < 64; d++) { const x = w[kb + d]; acc += a[qb + 2 * d] * H2F[x & 0xffff] + a[qb + 2 * d + 1] * H2F[x >>> 16]; }
          s = acc * 0.08838834764831845;
        }
        a[fi(S) + th * sStride + j] = s;
      }
    }
  },

  softmax(cpu, u, f, [S], [rows]) {
    const [L, stride] = u, a = cpu.f32();
    for (let r = 0; r < rows; r++) {
      const b = fi(S) + r * stride;
      let m = -3.0e38, s = 0;
      for (let j = 0; j < L; j++) if (a[b + j] > m) m = a[b + j];
      for (let j = 0; j < L; j++) { const e = a[b + j] <= -1.0e38 ? 0 : Math.exp(a[b + j] - m); a[b + j] = e; s += e; }
      for (let j = 0; j < L; j++) a[b + j] /= s;
    }
  },

  attnOutFor(cpu, u, f, bufs, [rows], [H, KVH, gated = true]) {
    const [S, VC] = bufs, G = gated ? bufs[2] : null, O = bufs[gated ? 3 : 2];
    const [L, sStride] = u, a = cpu.f32(), w = cpu.u32(), stride = KVH * 64, acc = new Float64Array(128);
    for (let th = 0; th < rows; th++) {
      const h = th % H, vbase = Math.floor(h / (H / KVH)) * 64;
      acc.fill(0);
      for (let j = 0; j < L; j++) {
        const p = a[fi(S) + th * sStride + j];
        if (!p) continue;
        const vb = fi(VC) + j * stride + vbase;
        for (let d = 0; d < 64; d++) { const x = w[vb + d]; acc[2 * d] += p * H2F[x & 0xffff]; acc[2 * d + 1] += p * H2F[x >>> 16]; }
      }
      const g = G ? sigmoid(a[fi(G) + th]) : 1;
      for (let d = 0; d < 128; d++) a[fi(O) + th * 128 + d] = acc[d] * g;
    }
  },

  attnFlashFor(cpu, u, f, bufs, groups, [H, KVH, gated = true]) {
    const [Q, KC, VC] = bufs, G = gated ? bufs[3] : null, O = bufs[gated ? 4 : 3];
    const [pos0, T, slots, L, window] = u, a = cpu.f32(), sc = scores(L);
    for (let th = 0; th < T * H; th++) {
      const t = Math.floor(th / H), h = th % H;
      const gate = G ? sigmoid(a[fi(G) + th]) : 1;
      attendRow(cpu, Q, fi(Q) + th * 128, KC, VC, Math.floor(h / (H / KVH)) * 128, KVH, L, slots, pos0 + T - 1, pos0 + t, window, O, fi(O) + th * 128, gate, sc);
    }
  },

  shortconv(cpu, u, f, [BCX, w, state, Y]) {
    const [T, N] = u, a = cpu.f32(), win = new Float32Array(7);
    for (let ch = 0; ch < N; ch++) {
      for (let k = 0; k < 6; k++) win[k] = a[fi(state) + k * N + ch];
      for (let t = 0; t < T; t++) {
        const row = fi(BCX) + t * 3 * N;
        win[6] = a[row + ch] * a[row + 2 * N + ch];
        let acc = 0;
        for (let k = 0; k < 7; k++) acc += win[k] * a[fi(w) + ch * 7 + k];
        a[fi(Y) + t * N + ch] = a[row + N + ch] * acc;
        win.copyWithin(0, 1);
      }
      for (let k = 0; k < 6; k++) a[fi(state) + k * N + ch] = win[k];
    }
  },

  swiglu(cpu, u, f, [G, U]) {
    const n = u[0], a = cpu.f32(), g0 = fi(G), u0 = fi(U);
    for (let i = 0; i < n; i++) { const g = a[g0 + i]; a[g0 + i] = (g / (1 + Math.exp(-g))) * a[u0 + i]; }
  },

  addInto(cpu, u, f, [X, Y]) {
    const n = u[0], a = cpu.f32(), x0 = fi(X), y0 = fi(Y);
    for (let i = 0; i < n; i++) a[y0 + i] += a[x0 + i];
  },

  topk16(cpu, u, f, [Lg, C], [slices]) {
    const n = u[0], a = cpu.f32(), w = cpu.u32(), cf = cpu.f32(), v = new Float32Array(256);
    for (let s = 0; s < slices; s++) {
      for (let i = 0; i < 256; i++) v[i] = s * 256 + i < n ? a[fi(Lg) + s * 256 + i] : -3.0e38;
      for (let r = 0; r < 16; r++) {
        let best = 0;
        for (let i = 1; i < 256; i++) if (v[i] > v[best]) best = i;
        cf[fi(C) + (s * 16 + r) * 2] = v[best];
        w[fi(C) + (s * 16 + r) * 2 + 1] = s * 256 + best;
        v[best] = -3.0e38;
      }
    }
  },
};
