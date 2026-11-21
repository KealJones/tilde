// Caches a byte source in fixed 16 MB pieces with the Cache API, so a model downloads once.
// Only one or two pieces are in memory at a time.

const PIECE = 16 << 20;

/**
 * @param {import('./gguf.js').ByteSource} src
 * @param {{ key: string, cacheName?: string, size?: number }} opts  `key` identifies the file (usually its URL).
 */
export async function cachedSource(src, { key, cacheName = 'tilde-v1', size } = {}) {
  if (typeof caches === 'undefined') return src;
  let cache;
  try { cache = await caches.open(cacheName); } catch { return src; }
  // The Cache API ignores #fragments when matching, so each piece gets its own path.
  const reqFor = (i) => new Request(`https://tilde.cache/${encodeURIComponent(key)}/${PIECE}/${i}`);
  const piece = async (i) => {
    const hit = await cache.match(reqFor(i));
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const len = size ? Math.min(PIECE, size - i * PIECE) : PIECE;
    const bytes = await readUpTo(src, i * PIECE, len);
    try { await cache.put(reqFor(i), new Response(bytes)); } catch { /* storage full: keep going uncached */ }
    return bytes;
  };
  return {
    name: src.name,
    async read(offset, length) {
      const first = Math.floor(offset / PIECE), last = Math.floor((offset + length - 1) / PIECE);
      if (first === last) {
        const p = await piece(first);
        return p.subarray(offset - first * PIECE, offset - first * PIECE + length);
      }
      const out = new Uint8Array(length);
      for (let i = first; i <= last; i++) {
        const p = await piece(i);
        const from = Math.max(offset, i * PIECE), to = Math.min(offset + length, i * PIECE + p.length);
        out.set(p.subarray(from - i * PIECE, to - i * PIECE), from - offset);
      }
      return out;
    },
  };
}

// The last piece of a file is short; a source may refuse to read past the end, so ask for less on failure.
async function readUpTo(src, offset, length, opts) {
  try { return await src.read(offset, length, opts); } catch (e) {
    if (!src.size || e.name === 'AbortError') throw e;
    return src.read(offset, Math.min(length, src.size - offset), opts);
  }
}

/** Deletes every cached model piece. */
export async function clearCache(cacheName = 'tilde-v1') {
  if (typeof caches !== 'undefined') await caches.delete(cacheName);
}

const sizeReq = (key) => new Request(`https://tilde.cache/${encodeURIComponent(key)}/size`);
const pieceReq = (key, i) => new Request(`https://tilde.cache/${encodeURIComponent(key)}/${PIECE}/${i}`);

/**
 * Downloads a whole file into the cache, piece by piece (resumable: pieces already there are skipped), so a later
 * load reads it from disk. `source` is a ByteSource for `url` (default: Range requests).
 * @param {string} url
 * `size`, when the caller knows it, saves asking the server (pieces a load already cached then need no network at all).
 * @param {{ key?: string, cacheName?: string, source?: import('./gguf.js').ByteSource, size?: number,
 *   onProgress?: (done: number, total: number) => void, signal?: AbortSignal }} [opts]
 */
export async function cacheFile(url, { key = url, cacheName = 'tilde-v1', source, size: known = 0, onProgress, signal } = {}) {
  const cache = await caches.open(cacheName);
  const src = source || (await import('./gguf.js')).httpSource(url);
  let size = Number(await (await cache.match(sizeReq(key)))?.text()) || known || 0;
  if (!size) { await src.read(0, 1, { signal }); size = src.size; }
  if (!size) throw new Error(`${url}: the server didn't say how big the file is`);
  const n = Math.ceil(size / PIECE);
  const lenOf = (i) => Math.min(PIECE, size - i * PIECE);
  let done = 0;
  for (let i = 0; i < n; i++) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    if (!(await cache.match(pieceReq(key, i)))) {
      // A server that ignores Range (src.rangeIgnored): the rest of the file in one request, cut into pieces as it
      // streams, instead of the whole file again for every piece.
      if (src.rangeIgnored && src.stream) { await streamRest(cache, key, src, i, n, lenOf, done, size, onProgress, signal); break; }
      const base = done;
      const bytes = await readUpTo(src, i * PIECE, lenOf(i), { signal, onBytes: (got) => onProgress?.(base + got, size) });
      await cache.put(pieceReq(key, i), new Response(bytes));
    }
    done += lenOf(i);
    onProgress?.(done, size);
  }
  await cache.put(sizeReq(key), new Response(String(size)));
}

// Pieces `from`… taken from one streamed response; pieces already cached are skipped over, not stored again.
async function streamRest(cache, key, src, from, n, lenOf, done, size, onProgress, signal) {
  const reader = await src.stream(from * PIECE, { signal });
  try {
    for (let i = from; i < n; i++) {
      const len = lenOf(i);
      const have = await cache.match(pieceReq(key, i));
      const buf = new Uint8Array(len);
      let got = 0;
      while (got < len) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const chunk = await reader.next(len - got);
        if (!chunk) throw new Error(`${key}: the download ended early at ${i * PIECE + got} of ${size} bytes`);
        if (!have) buf.set(chunk, got);
        got += chunk.length;
        onProgress?.(done + got, size);
      }
      if (!have) await cache.put(pieceReq(key, i), new Response(buf));
      done += len;
    }
  } finally { reader.cancel(); }
}

/**
 * How much of a file is cached: { size, stored, complete } (size 0 if it was never fully downloaded and no `size` is
 * given). Pieces cached by a load (cachedSource) carry no size record, so they count only when the caller says the size.
 */
export async function cachedFileInfo(key, cacheName = 'tilde-v1', { size: known = 0 } = {}) {
  if (typeof caches === 'undefined') return { size: 0, stored: 0, complete: false };
  const cache = await caches.open(cacheName);
  const size = Number(await (await cache.match(sizeReq(key)))?.text()) || known || 0;
  if (!size) return { size: 0, stored: 0, complete: false };
  let stored = 0;
  for (let i = 0; i < Math.ceil(size / PIECE); i++) if (await cache.match(pieceReq(key, i))) stored += Math.min(PIECE, size - i * PIECE);
  return { size, stored, complete: stored === size };
}

/** Deletes one file's pieces from the cache. */
export async function removeCachedFile(key, cacheName = 'tilde-v1') {
  if (typeof caches === 'undefined') return;
  const cache = await caches.open(cacheName);
  const prefix = `https://tilde.cache/${encodeURIComponent(key)}/`;
  for (const req of await cache.keys()) if (req.url.startsWith(prefix)) await cache.delete(req);
}
