# Tilde: notes for agents

WebGPU engine for small language models that must survive iPhone Safari (~1.9 GB per tab shared by JS, Wasm and
GPU; ~3 s per GPU command buffer). Read docs/design.md first; it is the contract.

## Rules

- Plain ES modules, no build step, JSDoc for types. Must run as-is from GitHub Pages.
- Never hold a whole model file in memory. Weights stream tensor by tensor through a small staging buffer.
- Every GPU buffer goes through the engine's allocator so memory is counted.
- GPU only. No CPU fallback path.
- Kernels must not need `shader-f16`: store halves in u32 and use `unpack2x16float`.
- Model-specific code lives in one file per architecture (src/models/<arch>.js).
- Every kernel change is checked against the CPU reference (src/quant.js, src/ref/) and, for whole-model changes,
  against llama.cpp logits for the same tokens.

## Commands

- `npm test`: CPU tests (GGUF parsing, repacking, reference math). The header test uses a local GGUF if
  `TILDE_GGUF` (or /home/claude/models/LFM3-1.7B-QAD-Q4_0.gguf) exists, otherwise it is skipped.
- `npm run gpu-test`: kernels in headless Chrome on SwiftShader WebGPU (needs Chrome; set CHROME to its path).

## Git

Push straight to main. Commits are authored as Keal Jones with Claude as co-author (Co-Authored-By trailer).

GitHub Pages serves the `gh-pages` branch: `.github/workflows/deploy.yml` copies main there on every push (after
`npm test`), and `pr-preview.yml` puts each PR at `https://kealjones.com/tilde/pr-preview/pr-<n>/` (also reachable
through the phone relay at `/tilde/pr-preview/pr-<n>/…`). Bigger or riskier changes go in a PR so they can be tried
on the phone first.

## Checking against llama.cpp

`bench/lfm3-check.html` runs a prompt through Tilde and compares the last-position logits and a greedy
continuation with a dump from llama.cpp (`[n][tokens][vocab][logits n×vocab][n_gen][gen]`, all little-endian
int32/f32). Mount the model and the dump into the bench server:

    TILDE_MOUNTS="/models=/path/to/ggufs /ref=/path/to/dumps" npm run gpu-test -- "lfm3-check.html?gen=4"

The page picks the model class from the GGUF's architecture (LFM3 or Qwen3). Make a dump with
`tools/llama-dump.cpp` (build line at its top): `llama-dump model.gguf "The capital of France is" out.bin 12`.

SwiftShader is slow (~80 s per token for LFM3 1.7B on 2 cores, ~17 s for Qwen3 0.6B), so keep `gen` small.
Baselines on "The capital of France is":
- LFM3 1.7B QAD Q4_0 (Nov 16, 2026): top 5 identical, max |logit diff| 0.50, 12/12 greedy tokens identical.
- Qwen3 0.6B Q4_0, ggml-org (Nov 18, 2026): top 5 identical, max |logit diff| 0.43, 8/8 greedy tokens identical.
- With the f16 KV cache (Nov 18, 2026): LFM3 0.494 and Qwen3 0.426, top 5 and greedy tokens still identical.
- 64×64 matmul tiles + fused prompt attention (Nov 19, 2026), 73–74-token ocean prompt fed in 64-token chunks
  (`?chunk=64&ref=/ref/lfm3-long.bin` or `qwen-long.bin`): LFM3 0.353 (3/3 greedy), Qwen3 0.460 (6/6 greedy), top 5 identical.
