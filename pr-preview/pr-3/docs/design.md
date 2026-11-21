# Tilde design

Tilde runs small language models in a browser tab on WebGPU, built to survive iPhone Safari. iOS kills a tab
that uses too much memory (Jetsam) or keeps the GPU busy too long; Tilde is built around both limits instead of
hoping to stay under them.

First model: **LFM3 1.7B** (Liquid AI). Next: **LFM3 Omni 2B**, with vision, hearing and speech swapped in and out
around a resident backbone.

## The limits we design for

| Limit | Where it bites | Measured on |
|---|---|---|
| ~1.9 GB per tab, shared by JS heap, WebAssembly and GPU buffers | loading weights, KV cache, activations | iPhone 16, Safari 27 (Nov 2026) |
| ~3 s per GPU command buffer, then WebKit kills the GPU process | long prefill, big matmuls | iPhone, Safari 27 |
| no `shader-f16` guaranteed (SwiftShader, some Android) | kernels | test bench |

The iPhone page and an iframe share the same 1.9 GB (tested with a cross-site iframe), so splitting work across
frames does not buy memory.

## Principles

1. **Never hold a whole file.** Weights arrive by HTTP range request, tensor by tensor, are repacked into the GPU
   layout in a small reused staging buffer, written to the GPU and dropped. Peak memory during load is the GPU
   weights so far plus one staging buffer, not the file twice.
2. **Account for every byte.** Every GPU buffer is allocated through one place that keeps a running total, so the
   engine can report what it holds and refuse a load that would not fit.
3. **Short GPU submits, sized by measurement.** Work is split into command buffers that each stay well under the
   3 s limit. The engine times each submit and splits finer when one runs long, instead of hard-coding Safari's
   number, because other browsers and drivers have their own undocumented limits.
4. **Only what's needed is resident.** For multimodal models the backbone stays loaded; each sense (vision encoder,
   audio encoder, speech generator) is loaded when needed and freed after.
5. **Download once.** URLs are cached in Cache Storage in 16 MB pieces (`src/cache.js`), each under its own path:
   the Cache API ignores `#fragments` when matching, so fragment-keyed pieces would all collide.
6. **GPU only.** No CPU fallback: a 1.7B model on a phone CPU takes minutes to answer "hi".
7. **No build step.** Plain ES modules with JSDoc types, served as-is (GitHub Pages works). Model-specific code
   lives in one file per architecture (LFM3, Qwen3), imported on demand by the GGUF's `general.architecture`; the
   attention kernels take the head counts as parameters, so a new GQA model reuses them.

## Model input

Official GGUF files, read directly. No conversion step, no extra hosting: Liquid publishes GGUFs for every LFM3
model, and the Omni encoders come as a separate `mmproj` GGUF.

`LFM3-1.7B-QAD-Q4_0.gguf` (1,050 MB, quantization-aware distilled):

| Tensors | Format |
|---|---|
| every matmul weight in the 24 layers | Q4_0 (32 values per block: f16 scale + 16 bytes of 4-bit values) |
| `token_embd` (128,000 × 2,048), also the output head | Q6_K (256 values per 210-byte block) |
| norms, conv kernels, `rope_freqs` | F32 |

GGUF blocks are 18 and 210 bytes, not 4-byte aligned, so the loader repacks them on the way in:
Q4_0 becomes a `u32` array of nibbles plus a `u32` array of f16 scale pairs; Q6_K is padded to 212 bytes per block.
Kernels read `u32` and use `unpack2x16float`, so `shader-f16` is optional.

## LFM3 backbone (from the GGUF metadata and llama.cpp's `lfm3.cpp`)

- 24 layers, hidden 2,048, FFN 7,680 (SwiGLU), vocab 128,000, RMSNorm eps 1e-5.
- Every layer: `x += post_norm(mixer(norm(x)))`, then `x += post_ffw_norm(ffn(ffn_norm(x)))` (sandwich norms).
- 13 **short-conv** layers (0, 1, 3, 5, …, 23): `in_proj` → B, C, X (2,048 each); `y = C ⊙ conv7(B ⊙ X)` (causal,
  depthwise, width 7, keeps the last 6 inputs as state); `out_proj`.
- 11 **attention** layers (2, 4, …, 22): 16 query heads, 4 KV heads, head dim 128, RMSNorm on q and k per head,
  NeoX RoPE, scale 1/√128, then a per-head **output gate** `sigmoid(W_gate x)` multiplied into each head's output,
  then `attn_output`.
  - 5 of them (4, 8, 12, 16, 20) are **sliding window** (256 tokens), RoPE base 10,000.
  - 6 are **global**, RoPE base 1,000,000 with `rope_freqs` factors (θ is divided by the factor).
- KV cache: only the 6 global layers grow with context (6 × 2 × 4 × 128 values per token, ~12 KB/token in f16:
  ~75 MB at 6K). The sliding-window layers keep a 256-token ring.
- Output: final RMSNorm, then the tied `token_embd` as the head.

## Memory plan (weights only)

| Piece | MB |
|---|---|
| LFM3 1.7B QAD Q4_0 layers | ~835 |
| token table / head (Q6_K) | ~215 |
| KV cache at 6K (f16) | ~75 |

KV entries are stored as f16 pairs in `u32` (llama.cpp's default type too): half of f32, which matters most for
models where every layer has a cache (Qwen3 0.6B: ~470 MB instead of ~940 MB at a 4K context). Later savings,
measured before adopting: English-only vocab trim (~160 MB off the table).

## Omni

`LFM3-Omni-2B` = the same backbone + `mmproj`: vision encoder + connector (~203 MB at Q8), audio encoder +
connector (~123 MB), speech generator `a.gen` (~181 MB). Peak with swapping is the backbone plus the largest single
sense (~1.3 GB of weights), instead of everything at once (~1.6 GB), which crashed Liquid's own WebGPU Space on an
iPhone at ~80% loaded.

- **Video** (`src/video.js`, `Tilde#readVideo`), as llama.cpp's `MTMD_VIDEO_TEMPLATE_LFM3_OMNI` reads it: 2 frames a
  second at `round(linspace(0, nSrc − 1, n))` of the source frames, n clamped to [4, 384]; frames go in pairs, one
  temporal patch each (the patchifier's two kernels: `W0·A + W1·B`; a still image is `W0·A + W1·A`). Each pair is
  `<t seconds>` (printf `%.1f`, ties to even), `<|vision_start|>` pair `<|vision_end|>`, then the pair's slice of the
  soundtrack, all inside one `<|vision_start|> … <|vision_end|>`. The soundtrack is encoded once as one clip and each
  pair takes rows `round(cumulative samples / 1280)`, the last pair the rest. Frames are read in the browser with
  `<video>` + canvas and scaled to `maxPixels` (256×256 → 64 tokens a pair); the vision encoder stays loaded for the
  clip.
- **Side sequence**: `LFM3#newSequence(ctx)` is a second set of KV and conv state (activations shared). `speak()` and
  `transcribe()` run there (2,048 positions, ~55 MB), so the conversation's prompt is still read afterwards.
- **Snapshots** (`generate({ snapshotAt })`): the conv states and sliding-window rings (~7 MB) are copied at a token
  count the next prompt is sure to share; a prompt that starts with those tokens goes back there and reads only the
  rest. Global KV slots up to that point are never written again, so they need no copy. This is what lets a chat
  template that drops the last turn's thinking, or a reply re-tokenized differently, still reuse the prompt.

## Testing

- `node --test`: GGUF parsing, repacking and CPU reference math.
- `npm run gpu-test`: kernels in headless Chrome on SwiftShader WebGPU, compared to the CPU reference and to
  llama.cpp's logits for the same prompt.
- Real devices: the phone relay (KealJones/phone-relay) drives iPhone Safari for load, memory and speed runs.

## Measured on an iPhone 16 (Safari 27, Nov 16, 2026)

| | LFM3 1.7B QAD Q4_0 | LFM3 Omni 2B Q4_0 + vision |
|---|---|---|
| Memory counted by the engine | 1,123 MB (ctx 4096) | 1,067 MB (ctx 2048); 1,280 MB peak while the vision encoder is swapped in |
| Load from Cache Storage | ~5 s | ~5 s (+0.4 s vision encoder) |
| Decode | 8 tok/s (56 ms GPU per token; the rest is CPU-side overhead) | 7 tok/s |
| Prompt reading | ~3–6 tok/s (naive batched matmul) | 64 image tokens in ~25 s |
| Vision encode, 256×256 image | – | ~1 s GPU, largest submit 442 ms |
| Correctness | answers correctly | 256×256 test image: first 8 greedy tokens identical to llama.cpp |

Lessons:
- 6 layers × 64 tokens in one submit (~4.5 s) made Safari drop the GPU. Submits are now sized ahead of time from
  per-batch-size estimates; cost per token is not flat in batch size (≈3.5 ms per layer-token at 8 tokens,
  ≈11.7 ms at 64), so one estimate per batch size.
- GitHub Pages lets Safari cache modules for 10 minutes: after a deploy, a reload can still run old code.

### After the speed pass (same phone, LFM3 1.7B, 256-token prompt, 64 generated tokens)

| | before | after |
|---|---|---|
| Prompt reading | 3–6 tok/s | **51 tok/s** (tiled 64×32 matmul) |
| Writing | 8 tok/s | **12.6 tok/s** |
| Per generated token | ~125 ms | 79.7 ms: 2.9 ms JS recording + 76.8 ms GPU, + 2.6 ms GPU top-k readback |
| Single-token matmul A/B | one workgroup per row: 100.6 ms/token | 8 rows per workgroup: 76.6 ms/token |

What made the difference: uniforms in slots of one buffer written once per submit (instead of a GPUBuffer per
dispatch), bind groups reused every decode step, GPU top-k (64 KB read back instead of 512 KB), and the tiled
prompt matmul. The rest of a decode step is GPU time: about 13 GB/s of weights read, so the kernels themselves
are the next lever.
