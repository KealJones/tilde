import { test } from 'node:test';
import assert from 'node:assert/strict';
import { alignIndex, harSource, istft, gaussian, KOKORO } from '../src/models/kokoro.js';

// Logits whose sigmoids sum to `s` exactly: k bins at +40 (1.0), the rest at -40 (~0), plus one bin at 0 (0.5).
const logitsFor = (s, nBins = 50) => { const l = new Float32Array(nBins).fill(-40); for (let i = 0; i < Math.floor(s); i++) l[i] = 40; if (s % 1) l[nBins - 1] = 0; return l; };

test('durations round half to even, at least 1, and the index repeats each token', () => {
  const l = new Float32Array([...logitsFor(2.5), ...logitsFor(3.5), ...logitsFor(0)]);
  const { durations, index } = alignIndex(l, 3, 50);
  assert.deepEqual([...durations], [2, 4, 1]);
  assert.deepEqual([...index], [0, 0, 1, 1, 1, 1, 2]);
  assert.deepEqual([...alignIndex(logitsFor(4), 1, 50, 2).durations], [2]);
});

test('the noise is seeded', () => {
  const a = gaussian(3), b = gaussian(3);
  const xs = Array.from({ length: 1000 }, () => a());
  assert.deepEqual(xs.slice(0, 5), Array.from({ length: 5 }, () => b()));
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length, sd = Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / xs.length);
  assert.ok(Math.abs(mean) < 0.1 && Math.abs(sd - 1) < 0.1);
});

test('the harmonic source of a 300 Hz tone peaks in the 300 Hz STFT bin', () => {
  const f0 = new Float32Array(16).fill(300), w = new Float32Array(9); w[0] = 10;
  const { har, nHar } = harSource(f0, w, [0], gaussian(1));
  assert.equal(nHar, 16 * KOKORO.UPS / KOKORO.HOP + 1);
  // bins are 1200 Hz apart (24 kHz / 20), so 300 Hz sits between bins 0 and 1; well inside, bin 0 and 1 beat the rest
  const row = har.subarray(40 * 22, 40 * 22 + 11);
  for (let k = 2; k < 11; k++) assert.ok(row[0] + row[1] > 4 * row[k], `bin ${k}`);
});

test('the inverse STFT has the right length and gives silence for a tiny spectrum', () => {
  const T = 9, spec = new Float32Array(T * 22).fill(-30);
  const out = istft(spec, T);
  assert.equal(out.length, (T - 1) * KOKORO.HOP);
  assert.ok(out.every((v) => Math.abs(v) < 1e-9));
});
