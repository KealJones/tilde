import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classify, nextLevels, backOff, fitMemory, BlackBox } from '../src/blackbox.js';

const mem = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, String(v)) }; };
const want = { ctx: 4096, chunk: 64, maxChunk: 256, submitBudgetMs: 2000 };

test('classify: crash, clean exit, discard, GPU loss', () => {
  assert.equal(classify(undefined), 'none');
  assert.equal(classify({ stage: 'reading' }), 'killed');
  assert.equal(classify({ stage: 'reading' }, true), 'discarded');
  assert.equal(classify({ closed: true }), 'clean');
  assert.equal(classify({ lost: true, closed: true }), 'gpu-lost');
});

test('nextLevels: deaths raise a level, three good runs give one back', () => {
  assert.deepEqual(nextLevels({}, 'killed'), { mem: 1, gpu: 0, good: 0 });
  assert.deepEqual(nextLevels({ mem: 3 }, 'killed'), { mem: 3, gpu: 0, good: 0 });
  assert.deepEqual(nextLevels({ mem: 1 }, 'gpu-lost'), { mem: 1, gpu: 1, good: 0 });
  let l = { mem: 1, gpu: 1, good: 0 };
  for (let i = 0; i < 2; i++) l = nextLevels(l, 'clean', { ran: true });
  assert.deepEqual(l, { mem: 1, gpu: 1, good: 2 });
  assert.deepEqual(nextLevels(l, 'clean', { ran: true }), { mem: 1, gpu: 0, good: 0 }); // GPU level first
  assert.deepEqual(nextLevels({ mem: 1, good: 2 }, 'clean', { ran: false }), { mem: 1, gpu: 0, good: 2 }); // didn't run: no credit
});

test('backOff: chunk halves, then context; budget shrinks 30% per level, with floors', () => {
  assert.deepEqual(backOff(want, { mem: 0, gpu: 0 }), want);
  assert.deepEqual(backOff(want, { mem: 1, gpu: 0 }), { ctx: 4096, chunk: 64, maxChunk: 128, submitBudgetMs: 2000 });
  assert.deepEqual(backOff(want, { mem: 2, gpu: 0 }), { ctx: 2048, chunk: 64, maxChunk: 64, submitBudgetMs: 2000 });
  assert.deepEqual(backOff(want, { mem: 3, gpu: 0 }), { ctx: 1024, chunk: 64, maxChunk: 64, submitBudgetMs: 2000 });
  assert.equal(backOff(want, { mem: 0, gpu: 2 }).submitBudgetMs, 980);
  assert.equal(backOff({ ...want, submitBudgetMs: 400 }, { mem: 0, gpu: 3 }).submitBudgetMs, 300);
  assert.equal(backOff({ ...want, maxChunk: 32, chunk: 32 }, { mem: 2, gpu: 0 }).maxChunk, 16);
});

test('BlackBox: a killed session backs off the next one; a clean one does not', () => {
  const storage = mem();
  const a = new BlackBox('m', want, { storage, target: null });
  assert.equal(a.verdict, 'none');
  a.note({ stage: 'reading', mb: 1065, ran: true });
  // tab killed here: no close()
  const b = new BlackBox('m', want, { storage, target: null });
  assert.equal(b.verdict, 'killed');
  assert.equal(b.previous.stage, 'reading');
  assert.equal(b.settings.maxChunk, 128);
  b.note({ ran: true }, true); b.close();
  const c = new BlackBox('m', want, { storage, target: null });
  assert.equal(c.verdict, 'clean');
  assert.equal(c.settings.maxChunk, 128); // one good run isn't enough to relax
  const d = new BlackBox('m', want, { storage, target: null, apply: false, wasDiscarded: true });
  assert.deepEqual(d.settings, want); // apply: false keeps the requested settings
});

test('a kill while hidden or just after coming back is not held against Tilde', () => {
  assert.equal(classify({ stage: 'gpu', hidden: true }), 'background');
  assert.equal(classify({ stage: 'gpu', hidden: false, shownAt: 1000, seen: 3000 }), 'background');
  assert.equal(classify({ stage: 'gpu', hidden: false, shownAt: 1000, seen: 9000 }), 'killed');
  assert.equal(classify({ lost: true, hidden: true }), 'background');
  assert.equal(classify({ hidden: true, closed: true }), 'clean');
  assert.deepEqual(nextLevels({ mem: 1 }, 'background', { ran: true }), { mem: 1, gpu: 0, good: 0 });
});

test('a kill after everything was allocated fits the next session under 85% of that memory', () => {
  // Qwen3 0.6B at a 16K context: ~430 MB weights, 1,792 MB KV, ~11 MB activations; died at 2,233 MB.
  const l = nextLevels({}, 'killed', { stage: 'gpu', mb: 2233 });
  assert.deepEqual(l, { mem: 0, gpu: 0, good: 0, capMB: 1898 });
  const split = { fixedMB: 430, kvMB: 1792, actMB: 11, ctx: 16384, maxChunk: 64 };
  const s = backOff({ ctx: 16384, chunk: 64, maxChunk: 64, submitBudgetMs: 1000 }, l, split);
  assert.equal(s.ctx, 8192); // the KV cache was the big part, so the context halves; chunks stay
  assert.equal(s.maxChunk, 64);
  // Activations-heavy: the chunk shrinks first.
  assert.deepEqual(fitMemory({ ctx: 4096, maxChunk: 1024 }, { fixedMB: 400, kvMB: 100, actMB: 900, ctx: 4096, maxChunk: 1024 }, 1000), { ctx: 4096, maxChunk: 512 });
  // Floors: never below a 1,024 context or 16-token chunks.
  assert.deepEqual(fitMemory({ ctx: 2048, maxChunk: 32 }, { fixedMB: 5000, kvMB: 10, actMB: 10, ctx: 2048, maxChunk: 32 }, 100), { ctx: 1024, maxChunk: 16 });
  // A kill while loading (memory not all allocated) falls back to levels.
  assert.deepEqual(nextLevels({}, 'killed', { stage: 'loading', mb: 865 }), { mem: 1, gpu: 0, good: 0 });
  // Good runs loosen the cap 10% at a time.
  let r = { mem: 0, gpu: 0, good: 0, capMB: 1898 };
  for (let i = 0; i < 3; i++) r = nextLevels(r, 'clean', { ran: true });
  assert.equal(r.capMB, 2088);
});
