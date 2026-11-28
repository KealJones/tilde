// Qwen3 (dense): pre-norm GQA attention with per-head RMSNorm on q and k, NeoX RoPE, SwiGLU FFN, Q4_0 or Q6_K
// embedding/head (or a separate head). Runs on src/models/decoder.js like LFM3. Matches llama.cpp's qwen3.cpp.

import { Decoder } from './decoder.js';

export class Qwen3 extends Decoder {
  static arch = 'qwen3';
  static layer = () => ['norm attn_norm', 'attention', 'add', 'norm ffn_norm', 'swiglu', 'add'];

  /** Layer weights must be Q4_0; the embedding and head Q4_0 or Q6_K. */
  static check(g) {
    for (const t of g.tensors) {
      const want = t.name === 'token_embd.weight' || t.name === 'output.weight' ? ['Q4_0', 'Q6_K'] : t.dims.length > 1 ? ['Q4_0'] : ['F32'];
      if (!want.includes(t.typeName)) throw new Error(`Tilde runs Qwen3 with ${want.join(' or ')} for ${t.name}; this file has ${t.typeName} (a pure Q4_0 GGUF works, e.g. ggml-org/Qwen3-0.6B-GGUF)`);
    }
  }

  /** Qwen3 has no media input. */
  async forwardEmbeds() { throw new Error('Qwen3 takes text only'); }
}
