// httpSource against a fake server: Hugging Face's redirect, Safari losing Range on it, servers that ignore Range,
// stalls and expired storage addresses (an iPhone 16, 2026-11-17: asked for 16 MB, got the whole 1 GB file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { httpSource } from '../src/gguf.js';

const FILE = new Uint8Array(100_000).map((_, i) => (i * 7) & 255);
const ORIGIN = 'https://huggingface.co/m/resolve/main/f.gguf';
const CDN = 'https://cdn.example/f.gguf?sig=1';

// A body that hands out `chunk` bytes at a time; `stallAfter` bytes in, it stops sending.
function body(bytes, { chunk = 4096, stallAfter = Infinity } = {}) {
  let at = 0, cancelled = false;
  return {
    sent: () => at, cancelled: () => cancelled,
    getReader: () => ({
      read: () => (cancelled ? Promise.resolve({ done: true })
        : at >= bytes.length ? Promise.resolve({ done: true })
        : at >= stallAfter ? new Promise(() => {})
        : Promise.resolve({ done: false, value: bytes.slice(at, (at += Math.min(chunk, bytes.length - at))) })),
      cancel: async () => { cancelled = true; },
    }),
  };
}

// A server: `rangeAt(url)` says whether Range is honoured for a request that ends at that address.
function server({ redirect = true, keepsRangeOnRedirect = false, rangeDirect = true, stalls = 0, expireAfter = Infinity } = {}) {
  const log = [];
  let direct = 0;
  const fetchImpl = async (url, { headers, signal }) => {
    const range = /bytes=(\d+)-(\d*)/.exec(headers.Range);
    const viaRedirect = url === ORIGIN && redirect;
    const end = viaRedirect ? CDN : url;
    if (url === CDN && ++direct > expireAfter) { log.push({ url, status: 403 }); return { status: 403, url, redirected: false, headers: new Map(), body: body(new Uint8Array()) }; }
    const honoured = viaRedirect ? keepsRangeOnRedirect : url === CDN ? rangeDirect : true;
    const from = Number(range[1]), to = range[2] ? Number(range[2]) : FILE.length - 1;
    const stall = log.filter(l => l.status).length < stalls;
    const b = honoured ? body(FILE.slice(from, Math.min(to + 1, FILE.length)), { stallAfter: stall ? 10 : Infinity }) : body(FILE, { stallAfter: stall ? 10 : Infinity });
    signal?.addEventListener('abort', () => b.getReader().cancel());
    const res = {
      status: honoured ? 206 : 200, url: end, redirected: viaRedirect,
      headers: new Map(honoured ? [['Content-Range', `bytes ${from}-${Math.min(to, FILE.length - 1)}/${FILE.length}`]] : [['Content-Length', String(FILE.length)]]),
      body: b,
    };
    res.headers.get = res.headers.get.bind(res.headers);
    log.push({ url, status: res.status, range: headers.Range, body: b });
    return res;
  };
  return { fetchImpl, log };
}

test('Range lost on the redirect (Safari): the read goes again straight to the storage server, and stays there', async () => {
  const s = server();
  const src = httpSource(ORIGIN, { fetchImpl: s.fetchImpl });
  assert.deepEqual(await src.read(5000, 100), FILE.slice(5000, 5100));
  assert.deepEqual(s.log.map(l => [l.url, l.status]), [[ORIGIN, 200], [CDN, 206]]);
  assert.ok(s.log[0].body.sent() < 10000, 'the whole-file answer was dropped, not read');
  assert.equal(src.size, FILE.length);
  assert.equal(src.rangeIgnored, false);
  await src.read(0, 10);
  assert.equal(s.log.at(-1).url, CDN, 'later reads skip the redirect');
});

test('a server that never honours Range: the part asked for is cut from the stream, never the whole file in memory', async () => {
  const s = server({ redirect: false });
  const fetchImpl = async (url, o) => { const r = await s.fetchImpl(url, { ...o, headers: { Range: 'bytes=0-' } }); return { ...r, status: 200 }; };
  const src = httpSource(ORIGIN, { fetchImpl });
  const seen = [];
  assert.deepEqual(await src.read(30000, 1000, { onBytes: g => seen.push(g) }), FILE.slice(30000, 31000));
  assert.equal(src.rangeIgnored, true);
  assert.equal(seen.at(-1), 1000);
  assert.ok(s.log[0].body.cancelled(), 'the rest of the file is not downloaded');
});

test('a transfer that goes quiet is dropped and asked again', async () => {
  const s = server({ redirect: false, stalls: 1 });
  const src = httpSource(ORIGIN, { fetchImpl: s.fetchImpl, stallMs: 50 });
  assert.deepEqual(await src.read(0, 20000), FILE.slice(0, 20000));
  assert.equal(s.log.length, 2);
  assert.ok(s.log[0].body.cancelled());
});

test('an expired storage address goes back to the original URL', async () => {
  const s = server({ keepsRangeOnRedirect: true, expireAfter: 1 });
  const src = httpSource(ORIGIN, { fetchImpl: s.fetchImpl });
  await src.read(0, 10);                 // origin → CDN (Range kept), now reads go to the CDN
  await src.read(10, 10);                // CDN, first direct read
  assert.deepEqual(await src.read(20, 10), FILE.slice(20, 30));   // CDN says 403 → origin again
  assert.deepEqual(s.log.map(l => [l.url, l.status]), [[ORIGIN, 206], [CDN, 206], [CDN, 403], [ORIGIN, 206]]);
});
