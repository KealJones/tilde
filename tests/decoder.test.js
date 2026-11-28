import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as K from '../src/kernels.js';

// The layer tables run on a recording stand-in for the GPU: no numbers, but every dispatch, so a table that names
// a missing weight or buffer fails here instead of on a phone. Numerics are checked against llama.cpp in
// bench/lfm3-check.html.
globalThis.GPUBufferUsage ??= { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 };

function mockGpu() {
  const kernels = [];
  const enc = () => ({ copyBufferToBuffer() {}, finish: () => ({}) });
  return {
    kernels,
    device: { queue: { writeBuffer() {}, submit() {} }, createCommandEncoder: enc },
    buffer: (size, tag) => ({ size, tag }),
    free() {},
    beginPlan() {},
    dispatch(e, code, bind) {
      for (const b of bind) if (b === undefined) throw new Error('dispatch with a missing buffer');
      kernels.push(code);
    },
    async run(fn, e) { fn(e || enc()); return 1; },
    async done() {},
    async read(b, n) { return new ArrayBuffer(n); },
  };
}

// A tiny GGUF header (zero weights) with LFM3's or Qwen3's tensor names.
function fakeGGUF(arch, { E, n, H, kv, swa }) {
  const tensors = [], size = { Q4_0: (c) => (c / 32) * 18, Q6_K: (c) => (c / 256) * 210, F32: (c) => c * 4 };
  let offset = 0;
  const t = (name, dims, typeName) => {
    const count = dims.reduce((a, b) => a * b), nbytes = size[typeName](count);
    tensors.push({ name, dims, count, typeName, offset, nbytes });
    offset += nbytes;
  };
  const F = 256, V = 256, kvAt = (i) => (Array.isArray(kv) ? kv[i] : kv);
  t('token_embd.weight', [E, V], 'Q6_K'); t('output_norm.weight', [E], 'F32');
  for (let i = 0; i < n; i++) {
    const p = `blk.${i}.`;
    for (const w of ['attn_norm', 'ffn_norm', ...(arch === 'lfm3' ? ['post_attention_norm', 'post_ffw_norm'] : [])]) t(p + w + '.weight', [E], 'F32');
    t(p + 'ffn_gate.weight', [E, F], 'Q4_0'); t(p + 'ffn_up.weight', [E, F], 'Q4_0'); t(p + 'ffn_down.weight', [F, E], 'Q4_0');
    if (kvAt(i)) {
      t(p + 'attn_q.weight', [E, H * 128], 'Q4_0'); t(p + 'attn_k.weight', [E, kvAt(i) * 128], 'Q4_0');
      t(p + 'attn_v.weight', [E, kvAt(i) * 128], 'Q4_0'); t(p + 'attn_output.weight', [H * 128, E], 'Q4_0');
      t(p + 'attn_q_norm.weight', [128], 'F32'); t(p + 'attn_k_norm.weight', [128], 'F32');
      if (arch === 'lfm3') t(p + 'attn_gate.weight', [E, H], 'Q4_0');
    } else {
      t(p + 'shortconv.in_proj.weight', [E, 3 * E], 'Q4_0'); t(p + 'shortconv.conv.weight', [3, E], 'F32');
      t(p + 'shortconv.out_proj.weight', [E, E], 'Q4_0');
    }
  }
  const k = (s) => `${arch}.${s}`;
  const kvs = {
    'general.architecture': arch, [k('block_count')]: n, [k('embedding_length')]: E, [k('feed_forward_length')]: F,
    [k('attention.layer_norm_rms_epsilon')]: 1e-5, [k('rope.freq_base')]: 1e6, [k('attention.head_count')]: H,
    [k('attention.head_count_kv')]: kv, 'tokenizer.ggml.tokens': Array.from({ length: V }, (_, i) => `t${i}`),
  };
  if (swa) Object.assign(kvs, { [k('rope.freq_base_swa')]: 1e4, [k('attention.sliding_window')]: 16, [k('attention.sliding_window_pattern')]: swa });
  if (arch === 'qwen3') kvs[k('attention.key_length')] = 128;
  return { kv: kvs, tensors };
}

const src = { read: async (o, n) => new Uint8Array(n) };
const count = (gpu, code) => gpu.kernels.filter((c) => c === code).length;

test('LFM3 runs its conv and gated attention layers from the table', async () => {
  const { LFM3 } = await import('../src/models/lfm3.js');
  const gpu = mockGpu();
  const m = await LFM3.load(gpu, src, { ctx: 64, chunk: 8, gguf: fakeGGUF('lfm3', { E: 2048, n: 4, H: 16, kv: [0, 4, 0, 4], swa: [0, 1, 0, 0] }) });
  assert.ok(m.a.bcx && m.a.gate);
  await m.forward([1, 2, 3]);
  await m.snapshot(); await m.forward([4]); await m.restore();
  assert.equal(count(gpu, K.shortconv), 4); // 2 conv layers × 2 forwards
  assert.equal(count(gpu, K.attnOutFor(16, 4, true)), 4);
});

test('Qwen3 runs pre-norm attention with no conv or gate buffers', async () => {
  const { Qwen3 } = await import('../src/models/qwen3.js');
  const gpu = mockGpu();
  const m = await Qwen3.load(gpu, src, { ctx: 64, chunk: 8, gguf: fakeGGUF('qwen3', { E: 1024, n: 2, H: 16, kv: 8 }) });
  assert.ok(!m.a.bcx && !m.a.gate);
  await m.forward([1, 2, 3]);
  assert.equal(count(gpu, K.attnOutFor(16, 8, false)), 2);
  assert.equal(count(gpu, K.addInto), 4);
  await assert.rejects(m.forwardEmbeds(), /text only/);
});

test('an unknown layer module fails at load, not mid-forward', async () => {
  const { Decoder } = await import('../src/models/decoder.js');
  class Bad extends Decoder { static arch = 'qwen3'; static layer = () => ['norm attn_norm', 'moe']; }
  await assert.rejects(Bad.load(mockGpu(), src, { gguf: fakeGGUF('qwen3', { E: 1024, n: 1, H: 16, kv: 8 }) }), /unknown layer module: moe/);
});
