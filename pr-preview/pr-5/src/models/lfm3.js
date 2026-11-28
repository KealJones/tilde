// LFM3 (Liquid AI): 24 layers of short conv / gated GQA attention (some sliding window), sandwich norms,
// SwiGLU FFN, tied Q6_K embedding/head. See docs/design.md for the math; the runtime is src/models/decoder.js.

import { Decoder } from './decoder.js';

export class LFM3 extends Decoder {
  static arch = 'lfm3';
  // x += post_norm(mixer(norm(x))), then x += post_ffw_norm(ffn(ffn_norm(x))). The mixer is attention on layers
  // the GGUF gives KV heads, short conv on the rest; the attention output gate comes from attn_gate.
  static layer = (L) => [
    'norm attn_norm', L.attn ? 'attention' : 'shortconv', 'add_norm post_attention_norm',
    'norm ffn_norm', 'swiglu', 'add_norm post_ffw_norm',
  ];
}
