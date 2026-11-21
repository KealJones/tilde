import test from 'node:test';
import assert from 'node:assert/strict';

// A fake Cache API that, like the real one, ignores #fragments when matching requests.
const store = new Map();
const keyOf = (r) => r.url.split('#')[0];
globalThis.caches = {
  open: async () => ({
    match: async (r) => (store.has(keyOf(r)) ? new Response(store.get(keyOf(r))) : undefined),
    put: async (r, res) => { store.set(keyOf(r), new Uint8Array(await res.arrayBuffer())); },
  }),
};
const { cachedSource, cacheFile, cachedFileInfo } = await import('../src/cache.js');

test('cached reads return the right bytes across 16 MB pieces, from network and from cache', async () => {
  const file = new Uint8Array(40 << 20);
  for (let i = 0; i < file.length; i += 4096) file[i] = (i >> 12) & 255;
  file.forEach((_, i) => { if (i % 4096) file[i] = i & 255; });
  let fetched = 0;
  const src = { name: 'f', read: async (o, n) => { const b = file.slice(o, Math.min(o + n, file.length)); fetched += b.length; return b; } };
  const ranges = [[0, 100], [16777200, 100], [5, 16 << 20], [30e6, 10e6], [33554431, 2], [(40 << 20) - 10, 10]];
  for (const pass of ['network', 'cache']) {
    const c = await cachedSource(src, { key: 'https://example.com/model.gguf' });
    for (const [o, n] of ranges) assert.deepEqual(await c.read(o, n), file.subarray(o, o + n), `${pass} ${o}+${n}`);
  }
  assert.ok(fetched <= 40 << 20, `downloaded ${fetched} bytes for a 40 MB file`);
});

test('pieces a load cached count once the size is known, and finishing them needs no network', async () => {
  store.clear();
  const size = (32 << 20) + 5;
  const file = new Uint8Array(size).map((_, i) => i & 255);
  let reads = 0;
  const src = { name: 'g', read: async (o, n) => { reads++; return file.slice(o, Math.min(o + n, size)); } };
  const key = 'https://example.com/g.gguf';
  const c = await cachedSource(src, { key, size });
  await c.read(0, size);                                   // a load streams the whole file through the cache
  assert.deepEqual(await cachedFileInfo(key), { size: 0, stored: 0, complete: false }, 'no size record: unknown');
  assert.deepEqual(await cachedFileInfo(key, 'tilde-v1', { size }), { size, stored: size, complete: true });
  reads = 0;
  const seen = [];
  await cacheFile(key, { source: src, size, onProgress: (d) => seen.push(d) });
  assert.equal(reads, 0, 'every piece was there and the size was given: nothing fetched');
  assert.equal(seen.at(-1), size);
  assert.deepEqual(await cachedFileInfo(key), { size, stored: size, complete: true }, 'the size record is written');
});

test('a server that ignores Range: cacheFile takes the rest of the file in one request, with progress inside pieces', async () => {
  store.clear();
  const size = (32 << 20) + 5;
  const file = new Uint8Array(size).map((_, i) => i & 255);
  let requests = 0;
  const src = {
    name: 's', size, rangeIgnored: true,
    read: async () => { throw new Error('should stream instead'); },
    stream: async (offset) => {
      requests++;
      let at = offset;
      return { next: async (max) => { if (at >= size) return null; const out = file.slice(at, Math.min(size, at + Math.min(max, 1 << 20))); at += out.length; return out; }, cancel() {} };
    },
  };
  const key = 'https://example.com/s.gguf';
  const seen = [];
  await cacheFile(key, { source: src, size, onProgress: (d) => seen.push(d) });
  assert.equal(requests, 1);
  assert.ok(seen.filter(d => d < 16 << 20).length > 5, 'progress moves inside the first piece');
  assert.equal(seen.at(-1), size);
  assert.deepEqual(await cachedFileInfo(key), { size, stored: size, complete: true });
});
