import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dequantQ4_0, dequantQ6_K, repackQ4_0, repackQ6_K, f16ToF32, f32ToF16 } from '../src/quant.js';

const fx = (f) => new URL(`./fixtures/${f}`, import.meta.url);
const bytes = (f) => new Uint8Array(readFileSync(fx(f)));
const floats = (f) => { const b = readFileSync(fx(f)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

test('Q4_0 dequant matches the gguf reference', () => {
  const got = dequantQ4_0(bytes('blk_0_ffn_gate.bin'));
  const want = floats('blk_0_ffn_gate.f32');
  assert.equal(got.length, want.length);
  for (let i = 0; i < want.length; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-6, `value ${i}: ${got[i]} vs ${want[i]}`);
});

test('Q6_K dequant matches the gguf reference', () => {
  const got = dequantQ6_K(bytes('token_embd.bin'));
  const want = floats('token_embd.f32');
  assert.equal(got.length, want.length);
  for (let i = 0; i < want.length; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-6, `value ${i}: ${got[i]} vs ${want[i]}`);
});

test('Q4_0 repack keeps every scale and nibble', () => {
  const src = bytes('blk_0_ffn_gate.bin');
  const { d, qs, blocks } = repackQ4_0(src);
  assert.equal(blocks, src.length / 18);
  const d16 = new Uint16Array(d.buffer), q8 = new Uint8Array(qs.buffer);
  for (let b = 0; b < blocks; b++) {
    assert.equal(d16[b], src[b * 18] | (src[b * 18 + 1] << 8));
    for (let j = 0; j < 16; j++) assert.equal(q8[b * 16 + j], src[b * 18 + 2 + j]);
  }
});

test('Q6_K repack pads blocks to 212 bytes', () => {
  const src = bytes('token_embd.bin');
  const { words, blocks } = repackQ6_K(src);
  assert.equal(words.length * 4, blocks * 212);
  const b8 = new Uint8Array(words.buffer);
  for (let b = 0; b < blocks; b++) for (let i = 0; i < 210; i++) assert.equal(b8[b * 212 + i], src[b * 210 + i]);
});

test('f16 round trip', () => {
  for (const v of [0, 1, -2.5, 65504, 6.1e-5, 1e-7, 0.333]) {
    const back = f16ToF32(f32ToF16(v));
    assert.ok(Math.abs(back - v) <= Math.abs(v) * 1e-3 + 6e-8, `${v} -> ${back}`);
  }
});

test('Q8_0 dequant and repack match the gguf reference', async () => {
  const { dequantQ8_0, repackQ8_0 } = await import('../src/quant.js');
  const src = bytes('v_ffn_up.bin');
  const got = dequantQ8_0(src), want = floats('v_ffn_up.f32');
  assert.equal(got.length, want.length);
  for (let i = 0; i < want.length; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-6, `value ${i}`);
  const { qs, d, blocks } = repackQ8_0(src);
  const q8 = new Uint8Array(qs.buffer), d16 = new Uint16Array(d.buffer);
  for (let b = 0; b < blocks; b++) {
    assert.equal(d16[b], src[b * 34] | (src[b * 34 + 1] << 8));
    for (let j = 0; j < 32; j++) assert.equal(q8[b * 32 + j], src[b * 34 + 2 + j]);
  }
});
