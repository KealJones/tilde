import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, openSync, readSync, statSync } from 'node:fs';
import { readGGUF, archKey } from '../src/gguf.js';

const MODEL = process.env.TILDE_GGUF || '/home/claude/models/LFM3-1.7B-QAD-Q4_0.gguf';
const info = JSON.parse(readFileSync(new URL('./fixtures/info.json', import.meta.url)));

function fileSource(path) {
  const fd = openSync(path, 'r');
  return { name: path, read: async (o, n) => { const b = Buffer.alloc(n); readSync(fd, b, 0, n, o); return new Uint8Array(b.buffer, b.byteOffset, n); } };
}

test('reads the LFM3 1.7B header without the whole file', { skip: !existsSync(MODEL) && 'model file not present' }, async () => {
  let read = 0;
  const src = fileSource(MODEL);
  const counting = { name: src.name, read: (o, n) => { read += n; return src.read(o, n); } };
  const g = await readGGUF(counting);
  assert.equal(g.kv['general.architecture'], 'lfm3');
  assert.equal(archKey(g, 'block_count'), 24);
  assert.equal(g.tensors.length, info.tensors);
  const last = g.tensors.find((t) => t.name === info.last.name);
  assert.equal(last.offset, info.last.offset);
  assert.equal(last.nbytes, info.last.nbytes);
  assert.equal(last.offset + last.nbytes, statSync(MODEL).size);
  assert.ok(read < 16 << 20, `header read used ${read} bytes`);
});
