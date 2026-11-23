// The CPU backend's kernels (src/cpu.js, src/cpu.c) against the reference math in src/quant.js, run through the
// same dispatch path the models use. With the model files present, also whole models against llama.cpp.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, openSync, readSync, readFileSync } from 'node:fs';
import { openCpu } from '../src/cpu.js';
import * as K from '../src/kernels.js';
import { repackQ4_0, repackQ6_K, dequantQ4_0, dequantQ6_K, f32ToF16 } from '../src/quant.js';
import { readGGUF } from '../src/gguf.js';

// Deterministic random numbers.
let seed = 1;
const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
const randn = (n, s = 1) => Float32Array.from({ length: n }, () => (rnd() * 2 - 1) * s);

// Random quant blocks with sane f16 scales.
function q4Blocks(n) {
  const b = new Uint8Array(n * 18);
  for (let i = 0; i < n; i++) {
    const h = f32ToF16(0.002 + rnd() * 0.02); b[i * 18] = h & 255; b[i * 18 + 1] = h >> 8;
    for (let j = 2; j < 18; j++) b[i * 18 + j] = (rnd() * 256) | 0;
  }
  return b;
}
function q6kBlocks(n) {
  const b = new Uint8Array(n * 210);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 208; j++) b[i * 210 + j] = (rnd() * 256) | 0;
    const h = f32ToF16(0.0005 + rnd() * 0.002); b[i * 210 + 208] = h & 255; b[i * 210 + 209] = h >> 8;
  }
  return b;
}

async function setup() {
  const cpu = await openCpu();
  const put = (data, tag = 'test') => { const b = cpu.buffer(data.byteLength, tag); cpu.device.queue.writeBuffer(b, 0, data); return b; };
  const run = (code, bindings, groups) => cpu.run((enc) => cpu.dispatch(enc, code, bindings, groups));
  const get = async (b, n) => new Float32Array(await cpu.read(b, n * 4));
  return { cpu, put, run, get };
}

function close(got, want, tol, what) {
  let worst = 0, scale = 0;
  for (let i = 0; i < want.length; i++) { worst = Math.max(worst, Math.abs(got[i] - want[i])); scale = Math.max(scale, Math.abs(want[i])); }
  assert.ok(worst <= tol * Math.max(1, scale), `${what}: max |diff| ${worst} (values up to ${scale})`);
}

test('Q4_0 matmul (decode, small and large batches, accumulate)', async () => {
  const { cpu, put, run, get } = await setup();
  const rows = 40, Kd = 256, raw = q4Blocks((rows * Kd) / 32), ref = dequantQ4_0(raw);
  const { d, qs } = repackQ4_0(raw);
  const W = cpu.buffer(qs.byteLength + d.byteLength, 'weights');
  cpu.device.queue.writeBuffer(W, 0, qs); cpu.device.queue.writeBuffer(W, qs.byteLength, d);
  for (const [T, code] of [[1, K.matvecQ4_0], [1, K.matmulQ4_0(1)], [5, K.gemmQ4_0], [40, K.gemmQ4_0x64], [3, K.matmulQ4_0(8)]]) {
    const X = randn(T * Kd), want = new Float32Array(T * rows);
    for (let t = 0; t < T; t++) for (let r = 0; r < rows; r++) { let s = 0; for (let k = 0; k < Kd; k++) s += ref[r * Kd + k] * X[t * Kd + k]; want[t * rows + r] = s; }
    const Y = put(new Float32Array(T * rows));
    await run(code, [Uint32Array.of(rows, Kd, T, qs.byteLength / 4, rows, 0, Kd, 0), W, put(X), Y], 1);
    close(await get(Y, T * rows), want, 0.02, `T=${T} ${K.kernelInfo.get(code)?.name}`);
  }
  // matmulQ4_0 with accumulate adds into Y.
  const X = randn(Kd), Y = put(new Float32Array(rows).fill(1));
  await run(K.matmulQ4_0(1), [Uint32Array.of(rows, Kd, 1, qs.byteLength / 4, rows, 0, Kd, 1), W, put(X), Y], 1);
  const want = new Float32Array(rows);
  for (let r = 0; r < rows; r++) { let s = 1; for (let k = 0; k < Kd; k++) s += ref[r * Kd + k] * X[k]; want[r] = s; }
  close(await get(Y, rows), want, 0.02, 'accumulate');
});

test('Q6_K head and embedding', async () => {
  const { cpu, put, run, get } = await setup();
  const rows = 24, Kd = 512, raw = q6kBlocks((rows * Kd) / 256), ref = dequantQ6_K(raw);
  const W = put(repackQ6_K(raw).words, 'weights');
  const X = randn(Kd + 7), xOff = 7, Y = put(new Float32Array(rows));
  await run(K.matmulQ6_K, [Uint32Array.of(rows, Kd, xOff, 0), W, put(X), Y], 1);
  const want = new Float32Array(rows);
  for (let r = 0; r < rows; r++) { let s = 0; for (let k = 0; k < Kd; k++) s += ref[r * Kd + k] * X[xOff + k]; want[r] = s; }
  close(await get(Y, rows), want, 0.02, 'matmulQ6_K');
  const tok = Uint32Array.of(3, 0, 23), E = put(new Float32Array(3 * Kd));
  await run(K.embedQ6_K, [Uint32Array.of(Kd, 3, 0, 0), W, put(tok), E], 3);
  const got = await get(E, 3 * Kd);
  for (let t = 0; t < 3; t++) close(got.subarray(t * Kd, (t + 1) * Kd), ref.subarray(tok[t] * Kd, (tok[t] + 1) * Kd), 1e-6, `embed row ${tok[t]}`);
});

test('short conv carries its state across calls', async () => {
  const { put, run, get } = await setup();
  const N = 8, T = 5, bcx = randn(T * 3 * N), w = randn(N * 7), state = put(new Float32Array(6 * N)), Y1 = put(new Float32Array(3 * N)), Y2 = put(new Float32Array(2 * N));
  // Whole sequence at once vs. 3 tokens then 2.
  const all = put(new Float32Array(T * N)), s0 = put(new Float32Array(6 * N));
  await run(K.shortconv, [Uint32Array.of(T, N, 0, 0), put(bcx), put(w), s0, all], 1);
  await run(K.shortconv, [Uint32Array.of(3, N, 0, 0), put(bcx.slice(0, 3 * 3 * N)), put(w), state, Y1], 1);
  await run(K.shortconv, [Uint32Array.of(2, N, 0, 0), put(bcx.slice(3 * 3 * N)), put(w), state, Y2], 1);
  const want = await get(all, T * N), a = await get(Y1, 3 * N), b = await get(Y2, 2 * N);
  close([...a, ...b], want, 1e-6, 'split conv');
  // And against the definition: y_t = C_t · Σ_k w[k] · (B⊙X)_{t-6+k}.
  for (let ch = 0; ch < N; ch++) for (let t = 0; t < T; t++) {
    let s = 0;
    for (let k = 0; k < 7; k++) { const tt = t - 6 + k; if (tt >= 0) s += w[ch * 7 + k] * bcx[tt * 3 * N + ch] * bcx[tt * 3 * N + 2 * N + ch]; }
    assert.ok(Math.abs(want[t * N + ch] - bcx[t * 3 * N + N + ch] * s) < 1e-5);
  }
});

test('attention: fused prompt path equals scores → softmax → out, gated and on a ring', async () => {
  const { cpu, put, run, get } = await setup();
  const H = 16, KVH = 4, T = 6, slots = 9, pos0 = 7, L = slots, win = 5;
  const kv = (n) => { const u = new Uint32Array(n); for (let i = 0; i < n; i++) u[i] = f32ToF16(rnd() - 0.5) | (f32ToF16(rnd() - 0.5) << 16); return u; };
  const KC = put(kv(slots * KVH * 64)), VC = put(kv(slots * KVH * 64)), Q = put(randn(T * H * 128)), G = put(randn(T * H));
  const flash = put(new Float32Array(T * H * 128)), slow = put(new Float32Array(T * H * 128)), S = put(new Float32Array(T * H * L));
  await run(K.attnFlash, [Uint32Array.of(pos0, T, slots, L, win, 0, 0, 0), Q, KC, VC, G, flash], 1);
  await run(K.attnScores, [Uint32Array.of(pos0, T, slots, L, win, L, 0, 0), Q, KC, S], 1);
  await run(K.softmax, [Uint32Array.of(L, L, 0, 0), S], T * H);
  await run(K.attnOut, [Uint32Array.of(L, L, 0, 0), S, VC, G, slow], T * H);
  close(await get(flash, T * H * 128), await get(slow, T * H * 128), 1e-5, 'flash vs slow');
  cpu.free(S);
});

test('buffers are counted, freed ranges reused', async () => {
  const { cpu } = await setup();
  const a = cpu.buffer(1000, 'x'), b = cpu.buffer(5000, 'x');
  assert.equal(cpu.report().byTag.x, 0);
  assert.equal(cpu.bytes.get('x'), 1008 + 5008);
  cpu.free(a);
  const c = cpu.buffer(900, 'y');
  assert.equal(c.off, a.off);
  cpu.free(b); cpu.free(c);
  assert.equal(cpu.total, 0);
});

// Whole models vs llama.cpp dumps (tools/llama-dump.cpp), when the files are here.
const MODELS = process.env.TILDE_MODELS || '/home/claude/models';
const REFS = process.env.TILDE_REFS || '/home/claude/ref';
for (const [file, ref] of [['Qwen3-0.6B-Q4_0.gguf', 'qwen-hello.bin'], ['LFM3-1.7B-QAD-Q4_0.gguf', 'lfm3-hello.bin']]) {
  const have = existsSync(`${MODELS}/${file}`) && existsSync(`${REFS}/${ref}`);
  test(`${file} on the CPU matches llama.cpp`, { skip: !have && 'model or reference dump not present' }, async () => {
    const fd = openSync(`${MODELS}/${file}`, 'r');
    const src = { name: file, read: async (o, n) => { const b = Buffer.alloc(n); readSync(fd, b, 0, n, o); return new Uint8Array(b.buffer, b.byteOffset, n); } };
    const buf = readFileSync(`${REFS}/${ref}`), ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), i = new Int32Array(ab);
    const n = i[0], toks = [...i.subarray(1, 1 + n)], nv = i[1 + n];
    const last = new Float32Array(ab, (2 + n) * 4, n * nv).subarray((n - 1) * nv);
    const refGen = [...i.subarray(3 + n + n * nv, 3 + n + n * nv + i[2 + n + n * nv])].slice(0, 6);
    const g = await readGGUF(src);
    const Model = g.kv['general.architecture'] === 'lfm3' ? (await import('../src/models/lfm3.js')).LFM3 : (await import('../src/models/qwen3.js')).Qwen3;
    const m = await Model.load(await openCpu(), src, { ctx: 256, gguf: g, submitBudgetMs: Infinity });
    await m.forward(toks);
    const argmax = (a) => { let b = 0; for (let k = 1; k < a.length; k++) if (a[k] > a[b]) b = k; return b; };
    let l = await m.logits(), maxd = 0;
    for (let k = 0; k < nv; k++) maxd = Math.max(maxd, Math.abs(l[k] - last[k]));
    assert.ok(maxd < 0.75, `max |logit diff| ${maxd}`); // LFM3 0.58, Qwen3 0.43 on Nov 20, 2026
    const gen = [];
    for (let k = 0; k < refGen.length; k++) { gen.push(argmax(l)); await m.forward([gen.at(-1)]); l = await m.logits(); }
    assert.deepEqual(gen, refGen);
  });
}
