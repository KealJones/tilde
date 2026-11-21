// GGUF reader that never holds the whole file: the header is read in growing chunks,
// tensor data is read later, one tensor (or part of one) at a time.

/** ggml tensor types we know the size of: [values per block, bytes per block]. */
export const GGML_TYPES = {
  0: { name: 'F32', block: 1, bytes: 4 },
  1: { name: 'F16', block: 1, bytes: 2 },
  2: { name: 'Q4_0', block: 32, bytes: 18 },
  3: { name: 'Q4_1', block: 32, bytes: 20 },
  8: { name: 'Q8_0', block: 32, bytes: 34 },
  12: { name: 'Q4_K', block: 256, bytes: 144 },
  13: { name: 'Q5_K', block: 256, bytes: 176 },
  14: { name: 'Q6_K', block: 256, bytes: 210 },
  30: { name: 'BF16', block: 1, bytes: 2 },
};

/**
 * A byte source: `read(offset, length)` resolves to exactly those bytes.
 * @typedef {{ read(offset: number, length: number): Promise<Uint8Array>, name?: string }} ByteSource
 */

/** Byte source over HTTP range requests. */
/**
 * Reads byte ranges of a file over HTTP.
 *
 * Hugging Face answers with a redirect to its storage server, and iPhone Safari loses the Range header on that
 * redirect (an iPhone 16, 2026-11-17: asked for 16 MB, got status 200 and the whole 1 GB file, which sat the
 * download at 0% and then got the tab killed). So: after the first answer, reads go straight to the address the
 * redirect ended at (`res.url`), where Range is kept; a signed address that expires (401/403/410) sends reads
 * back to the original URL. A server that still answers 200 is read as a stream, skipping to the part asked for
 * and stopping there, never as one buffer of the whole file; `rangeIgnored` is set so a caller copying the whole
 * file (cacheFile) can take it in one pass instead. A response that sends nothing for `stallMs` is dropped and
 * asked again, up to `retries` times.
 *
 * @param {string} url
 * @param {{ fetchImpl?: typeof fetch, stallMs?: number, retries?: number }} [opts]
 */
export function httpSource(url, { fetchImpl = fetch, stallMs = 30000, retries = 3 } = {}) {
  let target = url;
  const src = {
    name: url,
    size: 0,
    rangeIgnored: false,
    /**
     * @param {number} offset @param {number} length
     * @param {{ onBytes?: (got: number) => void, signal?: AbortSignal }} [o] onBytes: bytes of this read so far
     */
    async read(offset, length, { onBytes, signal } = {}) {
      let lastError;
      for (let attempt = 0; attempt <= retries; attempt++) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        if (attempt) await new Promise(r => setTimeout(r, Math.min(8000, 500 * 2 ** attempt)));
        try {
          return await once(offset, length, onBytes, signal);
        } catch (e) {
          if (e.name === 'AbortError' && signal?.aborted) throw e;
          if (e.fatal) throw e;
          lastError = e;
          if (e.expired) target = url;
        }
      }
      throw lastError;
    },
    /** One request from `offset` to the end, read as a stream (for servers that ignore Range). */
    async stream(offset, { signal } = {}) {
      const res = await request(offset, null, signal);
      return skipTo(res.body.getReader(), res.status === 206 ? 0 : offset, stallMs, res.abort);
    },
  };

  async function request(offset, length, signal) {
    const ctl = new AbortController();
    const stop = () => ctl.abort();
    signal?.addEventListener('abort', stop, { once: true });
    const range = length == null ? `bytes=${offset}-` : `bytes=${offset}-${offset + length - 1}`;
    const timer = setTimeout(stop, stallMs);
    let res;
    try {
      res = await fetchImpl(target, { headers: { Range: range }, signal: ctl.signal });
    } catch (e) {
      throw ctl.signal.aborted && !signal?.aborted ? Object.assign(new Error(`${url}: no answer in ${stallMs / 1000} s`), { name: 'StallError' }) : e;
    } finally { clearTimeout(timer); }
    if ([401, 403, 410].includes(res.status) && target !== url) throw Object.assign(new Error(`${url}: the storage address expired (${res.status})`), { expired: true });
    if (res.status !== 206 && res.status !== 200) throw Object.assign(new Error(`GGUF read ${res.status} for ${url}`), { fatal: res.status === 404 });
    // Straight to where the redirect ended from now on: Range survives there.
    if (res.url && res.url !== target) target = res.url;
    const total = Number(/\/(\d+)$/.exec(res.headers.get('Content-Range') || '')?.[1]) || (res.status === 200 ? Number(res.headers.get('Content-Length')) : 0);
    if (total) src.size = total;
    if (res.status === 200 && !res.redirected) src.rangeIgnored = true;
    res.abort = stop;   // ends the transfer: a body left unread would otherwise keep downloading
    return res;
  }

  async function once(offset, length, onBytes, signal) {
    let res = await request(offset, length, signal);
    // The Range was lost on the redirect: ask the storage server itself, once, before reading a whole file.
    if (res.status === 200 && res.redirected && res.url) {
      res.abort();
      res = await request(offset, length, signal);
    }
    const reader = skipTo(res.body.getReader(), res.status === 206 ? 0 : offset, stallMs, res.abort);
    try {
      return await readInto(reader, offset, length, onBytes);
    } finally { reader.cancel(); }
  }

  async function readInto(reader, offset, length, onBytes) {
    const want = Math.min(length, src.size ? src.size - offset : length);
    const out = new Uint8Array(want);
    let got = 0;
    while (got < want) {
      const chunk = await reader.next(want - got);
      if (!chunk) break;
      out.set(chunk, got);
      got += chunk.length;
      onBytes?.(got);
    }
    // Reads that run past the end of the file come back short; that's fine at the end.
    if (got !== length && !(src.size && offset + got === src.size)) {
      throw new Error(`GGUF short read: wanted ${length} bytes at ${offset}, got ${got}`);
    }
    return got === want ? out : out.subarray(0, got);
  }
  return src;
}

// A response body read in chunks: the first `skip` bytes dropped, at most `max` bytes per call, and a chunk that
// takes longer than `stallMs` to arrive is an error (the read is then asked again).
function skipTo(reader, skip, stallMs, abort = () => {}) {
  let pending = null;
  const pull = async () => {
    let timer;
    const stalled = new Promise((_, no) => { timer = setTimeout(() => no(Object.assign(new Error(`download stalled for ${stallMs / 1000} s`), { name: 'StallError' })), stallMs); });
    try { return await Promise.race([reader.read(), stalled]); } catch (e) { abort(); throw e; } finally { clearTimeout(timer); }
  };
  return {
    async next(max = Infinity) {
      for (;;) {
        if (!pending) {
          const { done, value } = await pull();
          if (done) return null;
          pending = value;
        }
        if (skip >= pending.length) { skip -= pending.length; pending = null; continue; }
        if (skip) { pending = pending.subarray(skip); skip = 0; }
        const out = pending.length > max ? pending.subarray(0, max) : pending;
        pending = out === pending ? null : pending.subarray(max);
        return out;
      }
    },
    cancel() { pending = null; abort(); reader.cancel().catch(() => {}); },
  };
}

/** Byte source over an in-memory buffer (tests). */
export function bufferSource(bytes) {
  return { name: 'buffer', read: async (o, n) => bytes.subarray(o, o + n) };
}

const DEC = new TextDecoder();

/**
 * Parses a GGUF header.
 * @param {ByteSource} src
 * @returns {Promise<{ version: number, kv: Record<string, any>, tensors: Tensor[], dataOffset: number, alignment: number }>}
 * @typedef {{ name: string, dims: number[], type: number, typeName: string, count: number, offset: number, nbytes: number }} Tensor
 */
export async function readGGUF(src, { chunk = 4 << 20 } = {}) {
  let buf = await src.read(0, chunk);
  let end = buf.length;
  let p = 0;

  // Reads past the buffer fetch the next chunk and retry.
  const need = async (n) => {
    while (p + n > end) {
      const more = await src.read(end, Math.max(chunk, n));
      const grown = new Uint8Array(end + more.length);
      grown.set(buf.subarray(0, end));
      grown.set(more, end);
      buf = grown;
      end = grown.length;
    }
  };
  const view = () => new DataView(buf.buffer, buf.byteOffset, end);
  const u32 = async () => { await need(4); const v = view().getUint32(p, true); p += 4; return v; };
  const u64 = async () => {
    await need(8);
    const v = view().getBigUint64(p, true); p += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('GGUF value too large');
    return Number(v);
  };
  const str = async () => { const n = await u64(); await need(n); const s = DEC.decode(buf.subarray(p, p + n)); p += n; return s; };

  const scalar = {
    0: [1, (d, o) => d.getUint8(o)], 1: [1, (d, o) => d.getInt8(o)],
    2: [2, (d, o) => d.getUint16(o, true)], 3: [2, (d, o) => d.getInt16(o, true)],
    4: [4, (d, o) => d.getUint32(o, true)], 5: [4, (d, o) => d.getInt32(o, true)],
    6: [4, (d, o) => d.getFloat32(o, true)], 7: [1, (d, o) => d.getUint8(o) !== 0],
    10: [8, (d, o) => Number(d.getBigUint64(o, true))], 11: [8, (d, o) => Number(d.getBigInt64(o, true))],
    12: [8, (d, o) => d.getFloat64(o, true)],
  };
  const value = async (t) => {
    if (t === 8) return str();
    if (t === 9) {
      const et = await u32();
      const n = await u64();
      if (et === 8) { const out = new Array(n); for (let i = 0; i < n; i++) out[i] = await str(); return out; }
      const [size, get] = scalar[et] || [];
      if (!size) throw new Error(`GGUF array of unknown type ${et}`);
      await need(size * n);
      const d = view(); const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = get(d, p + i * size);
      p += size * n;
      return out;
    }
    const [size, get] = scalar[t] || [];
    if (!size) throw new Error(`GGUF value of unknown type ${t}`);
    await need(size); const v = get(view(), p); p += size; return v;
  };

  await need(4);
  if (DEC.decode(buf.subarray(0, 4)) !== 'GGUF') throw new Error(`${src.name || 'file'} is not a GGUF file`);
  p = 4;
  const version = await u32();
  if (version < 2) throw new Error(`GGUF version ${version} is not supported`);
  const nTensors = await u64();
  const nKV = await u64();

  const kv = {};
  for (let i = 0; i < nKV; i++) { const k = await str(); kv[k] = await value(await u32()); }

  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = await str();
    const nd = await u32();
    const dims = [];
    for (let j = 0; j < nd; j++) dims.push(await u64());
    const type = await u32();
    const rel = await u64();
    const info = GGML_TYPES[type];
    const count = dims.reduce((a, b) => a * b, 1);
    tensors.push({
      name, dims, type, typeName: info ? info.name : `type${type}`, count, offset: rel,
      nbytes: info ? (count / info.block) * info.bytes : NaN,
    });
  }

  const alignment = kv['general.alignment'] || 32;
  const dataOffset = Math.ceil(p / alignment) * alignment;
  for (const t of tensors) t.offset += dataOffset;
  return { version, kv, tensors, dataOffset, alignment };
}

/** Architecture-prefixed key, e.g. arch('lfm3', 'block_count'). */
export const archKey = (gguf, key) => gguf.kv[`${gguf.kv['general.architecture']}.${key}`];
