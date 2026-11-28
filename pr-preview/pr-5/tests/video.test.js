// Video sampling against llama.cpp (mtmd-helper.cpp, MTMD_VIDEO_TEMPLATE_LFM3_OMNI) on bench's 4 s, 10 fps clip:
// frames at round(linspace(0, 39, 8)), a timestamp per pair, the soundtrack split per pair (14, 14, 13, 9 rows).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { frameTimes, groups, stamp } from '../src/video.js';

test('frames: 2 a second, spread over the source frames, at least 4', () => {
  const { times } = frameTimes(4, { srcFps: 10 });
  assert.deepEqual(times, [0, 0.6, 1.1, 1.7, 2.2, 2.8, 3.3, 3.9]);
  assert.equal(frameTimes(1, { srcFps: 30 }).times.length, 4, 'short clips still get 4 frames');
  assert.equal(frameTimes(600, { maxFrames: 384 }).times.length, 384);
});

test('pairs: timestamp text as printf("%.1f") writes it, and the soundtrack slice of each pair', () => {
  const g = groups(frameTimes(4, { srcFps: 10 }).times, 64000);
  assert.deepEqual(g.map((x) => x.text), ['<0.3 seconds>', '<1.4 seconds>', '<2.5 seconds>', '<3.6 seconds>']);
  assert.deepEqual(g.map((x) => x.frames), [[0, 1], [2, 3], [4, 5], [6, 7]]);
  // Rows per pair the way index.js assigns them: round-half-even(cumulative samples / 1280), the last takes the rest.
  const rows = []; let s = 0, row = 0;
  g.forEach((x, i) => {
    s += x.audio[1] - x.audio[0];
    const y = s / 1280, r = Math.round(y), even = Math.abs(y % 1) === 0.5 && r % 2 ? r - 1 : r;
    const end = i === g.length - 1 ? 50 : Math.max(row, even);
    rows.push(end - row); row = end;
  });
  assert.deepEqual(rows, [14, 14, 13, 9]);
  assert.equal(stamp(0.25), '<0.2 seconds>', 'an exact tie goes to even, as in C');
  assert.equal(stamp(0.75), '<0.8 seconds>');
  const odd = groups([0, 0.5, 1], 0);
  assert.deepEqual(odd.at(-1).frames, [2, 2], 'a short last pair repeats its frame');
});
