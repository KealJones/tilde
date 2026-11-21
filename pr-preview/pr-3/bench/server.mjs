// Static file server with HTTP Range support (python's http.server has none). Usage:
//   node bench/server.mjs [port] [/url-prefix=dir ...]   (repo root is served at /)
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { join, extname, resolve } from 'node:path';

const main = import.meta.url === `file://${process.argv[1]}`;
const port = main ? Number(process.argv[2] || 8790) : 0;
const mounts = [['/', resolve(new URL('..', import.meta.url).pathname)]];
for (const a of [...(main ? process.argv.slice(3) : []), ...(process.env.TILDE_MOUNTS || '').split(' ').filter(Boolean)]) { const [p, d] = a.split('='); mounts.unshift([p.endsWith('/') ? p : p + '/', resolve(d)]); }
const TYPES = { '.mp4': 'video/mp4', '.webm': 'video/webm', '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };

export function serve() {
  return createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const [prefix, dir] = mounts.find(([p]) => url.startsWith(p));
    const file = join(dir, url.slice(prefix.length));
    if (!file.startsWith(dir)) { res.writeHead(403).end(); return; }
    let st;
    try { st = statSync(file); if (st.isDirectory()) throw 0; } catch { res.writeHead(404).end(); return; }
    const headers = { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store' };
    const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range || '');
    if (m) {
      const start = Number(m[1]), end = m[2] ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
      res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${st.size}`, 'Content-Length': end - start + 1 });
      createReadStream(file, { start, end }).pipe(res);
    } else {
      res.writeHead(200, { ...headers, 'Content-Length': st.size });
      createReadStream(file).pipe(res);
    }
  }).listen(port);
}

if (main) { serve(); console.log(`serving on http://localhost:${port}`); }
