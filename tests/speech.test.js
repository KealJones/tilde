import { test } from 'node:test';
import assert from 'node:assert/strict';
import { trimSilence, chunkSentences } from '../src/speech.js';

test('trimSilence keeps 30 ms before the sound and 250 ms after it', () => {
  const pcm = new Float32Array(24000); // 1 s: silence, a tone from 0.3 s to 0.5 s, silence
  pcm.fill(0.5, 7200, 12000);
  const out = trimSilence(pcm);
  assert.equal(out.length, (12000 - 7200) + 720 + 6000);
  assert.equal(out[720], pcm[7200]);
  assert.equal(trimSilence(new Float32Array(10)).length, 10); // all silence: nothing to cut against
});

test('chunkSentences starts with one sentence and grows each chunk up to 3× the one before', () => {
  const s = (n) => 'x'.repeat(n);
  const out = chunkSentences([s(20), s(30), s(40), s(40), s(100), s(300)], { max: 250 });
  assert.deepEqual(out.map((c) => c.length), [20, 30, 81, 100, 300]);
  assert.deepEqual(chunkSentences(['a', 'b']), ['a', 'b']);
});
