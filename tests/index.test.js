import { test } from 'node:test';
import assert from 'node:assert/strict';

// An export that index.js never imports is a link error: the whole module fails to load in every browser.
test('index.js loads and exports the speech helpers', async () => {
  const m = await import('../src/index.js');
  for (const k of ['loadKokoro', 'trimSilence', 'chunkSentences', 'ClipPlayer', 'cacheFile']) assert.equal(typeof m[k], 'function', k);
});
