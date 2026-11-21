# Tilde next to the other browser runtimes

Every runtime below ran the same model, Liquid AI's **LFM3 1.7B** (4-bit), in Safari on one iPhone 16, Nov 2026.
The rows link to where each result is recorded. [versus.md](versus.md) explains, with diagrams, why each runtime
breaks where it does. Results that were seen but never written down are left out on
purpose; [Next runs](#next-runs) lists what would fill the gaps.

## Head to head: one phone, one day (Nov 18, 2026)

[`bench/compare.html`](../bench/compare.html) on an iPhone 16, Safari 27.2: every runtime in a fresh page load,
the same text, greedy, 64 tokens written, 4K context, one warm-up request each. One run per cell.

| Runtime | 256-token prompt | 1,024 | 3,072 |
|---|---|---|---|
| **Tilde** | ✅ read 50, write 12.0, first token 5.1 s | ✅ read 34.9, write 6.9, first token 29.4 s | ✅ read 40.8, write 8.6, first token 75.3 s |
| **WebLLM 0.2.87** (our QAD MLC build), stock | ✅ read 84.4, write 12.3, first token 3.0 s | 💀 tab killed while loading | skipped |
| **WebLLM + Squiggle's pacing** | ✅ read 75.5, write 12.1, first token 3.4 s | ✅ read 66.7, write 8.4, first token 15.4 s | 💀 tab killed while loading |
| **wllama 3.13 WebGPU** (Safari build) | 💀 tab killed while loading | skipped | skipped |
| **wllama 3.13 CPU** (Safari build, 1 thread) | ✅ read 1.8, write 1.4, first token 2.4 min | ✅ read 1.6, write 1.6, first token 10.8 min | ⏱ no answer in 25 min |
| **transformers.js 4.4** (onnx-community q4f16) | 💀 tab killed while loading the model, twice | skipped | skipped |
| **LiteRT-LM 0.19** (litert-community int4) | 💀 tab killed while loading, at 1,593 MB of GPU buffers | skipped | skipped |

Speeds in tokens per second. GPU memory: Tilde 1,124 MB, WebLLM 1,040 MB (both counted by watching
`createBuffer`). All three GPU runtimes that answered wrote the same continuation.

What this run says, and doesn't:

- **Only Tilde finished every prompt size.** That is the whole of its lead in this run.
- **Where WebLLM survived, it was faster.** With Squiggle's pacing it read prompts 1.5× (256) to 1.9× (1,024) faster
  and reached the first token in about half the time. Writing speed was about even at 256 and in WebLLM's favour at
  1,024.
- **Both WebLLM deaths happened while loading,** before any prompt was read, so they are not caused by the prompt
  size. They may come from memory left behind by the test before (iOS frees a closed page's memory lazily) and need
  reruns before they count against WebLLM.
- **The 1,024 round was slow for everyone.** Tilde read and wrote more slowly at 1,024 than at 3,072, which a longer
  context can't explain, and paced WebLLM dropped in the same round; it came right after 3 minutes of wllama on the
  CPU. Repeats and a fixed GPU workload timed before each test would separate heat from engine speed.
- **Stock WebLLM survived a 256-token prompt** on Safari 27.2, so the watchdog bites somewhere above that, not on
  every prompt.

Full table with every note: [runs/2026-11-18-iphone16-lfm3.md](runs/2026-11-18-iphone16-lfm3.md).

**After the kernel rewrite (Nov 19 afternoon, [runs](runs/2026-11-19-iphone16-lfm3-kernels.md)):** over two runs,
Tilde reads LFM3 prompts at 138–139 / 137 / 119 tok/s by default (256 / 1,024 / 3,072 tokens) and 157 / 152–157 /
125–144 with 256-token chunks: 2.8–4.5× faster than Nov 18 and faster than WebLLM's 84 (stock, 256) and 67 (paced,
1,024). First token on a 3,072-token prompt: 75 s → 21–26 s. Writing is about unchanged (12 tok/s at 256). One
default run was killed while loading, below its own steady memory; the rerun finished every size.

**Tilde and stock WebLLM taking turns (Nov 19, normal power, [run](runs/2026-11-19-iphone16-lfm3-vs-webllm.md)):** at
256 tokens Tilde (256-token chunks) read 1.45× faster than WebLLM (149 against 102.5 tok/s) and reached the first token
sooner (1.7 s against 2.5), while WebLLM wrote 5% faster (12.7 against 12.1). WebLLM was then killed while loading the
1,024 run; Tilde finished 1,024 (149 / 11.4) and 3,072 (115 / 8.9). An earlier WebLLM run that afternoon
([run](runs/2026-11-19-iphone16-lfm3-webllm.md)) was on Low Power Mode and doesn't count.

**Tilde and paced WebLLM taking turns (Nov 19, normal power, [run](runs/2026-11-19-iphone16-lfm3-vs-webllm-paced.md)):**
both finished every size. Tilde (256-token chunks) read 1.57× (256), 1.63× (1,024) and 1.75× (3,072) faster — 155 /
156 / 141 against 99 / 96 / 80 tok/s — and reached the first token in 21.8 s against 38.2 s at 3,072. Writing was
even (12.3 / 12.0 / 10.7 against 12.5 / 11.8 / 10.2). WebLLM loaded faster (~4 s against ~5.6 s) in ~50 MB less
memory.

## Qwen3 0.6B: one phone, two passes (Nov 19, 2026)

The same bench with a model a third the size. Tilde and wllama read ggml-org's pure Q4_0 GGUF; WebLLM ran its own
prebuilt q4f16_1 build, transformers.js onnx-community's q4f16 and LiteRT-LM litert-community's 4-bit build. The
first pass ([export](runs/2026-11-19-iphone16-qwen3.md)) had Tilde with an f32 KV cache and lost transformers.js and
LiteRT-LM to a locked phone; the second pass ([export](runs/2026-11-19-iphone16-qwen3-pass2.md)) reran those with
Tilde's f16 cache. That afternoon Tilde got new matmul and attention kernels and ran again
([export](runs/2026-11-19-iphone16-qwen3-kernels.md)). One run per cell; read / write in tokens per second.

| Runtime | Memory | 256-token prompt | 1,024 | 3,072 |
|---|---:|---|---|---|
| **Tilde**, new kernels (afternoon) | **865 MB** | ✅ 322 / 26.7 | ✅ 268 / 21.0 | ✅ 178 / 14.5 |
| **Tilde**, new kernels, 256-token chunks | 879 MB | ✅ 407 / 27.1 | ✅ 337 / 21.5 | ✅ 214 / 14.4 |
| Tilde, f16 KV (pass 2) | 873 MB | ✅ 134 / 26.9 | ✅ 116 / 21.7 | ✅ 83 / 14.6 |
| Tilde, f32 KV (pass 1) | 1,321 MB | ✅ 128 / 25.3 | ✅ 109 / 20.2 | ✅ 62 / 11.1 |
| **LiteRT-LM 0.19** (pass 2) | 1,496 MB | ✅ **610 / 37.8** | ✅ **803 / 33.4** | 💀 lost the GPU while reading |
| **WebLLM 0.2.87**, stock | 1,125 MB | ✅ 215 / 31.3 | 💀 lost the GPU while reading | skipped |
| **WebLLM + Squiggle's pacing** | 1,129 MB | ✅ 181 / 19.8 | ✅ 205 / 19.3 | 💀 lost the GPU while reading |
| **wllama 3.13 WebGPU** | – | ❌ aborted while running (both passes) | skipped | skipped |
| **wllama 3.13 CPU** | – | ✅ 6.1 / 2.3 | ✅ 5.0 / 2.2 (3.5 min to first token) | 💀 killed (page hidden part of the time) |
| **transformers.js 4.4** | – | 💀 killed while downloading or loading (pass 2) | skipped | skipped |

- **LiteRT-LM is the fastest runtime here,** where it survives. Against Tilde's morning build it read prompts 4.5–7×
  faster; against the afternoon's new kernels the gap is 1.5–3× (2.4× at 1,024 tokens with 256-token chunks), and its
  first token of a 1,024-token prompt comes in 1.3 s against Tilde's 3.0. It also writes 1.3–1.6× faster. Its
  speeds are its own benchmark counts, and its prompt includes the chat template's few extra tokens.
- **Tilde now reads prompts faster than WebLLM:** 322 tok/s at 256 tokens against stock WebLLM's 215, and 268 at
  1,024 against paced WebLLM's 205 (different sessions of the same day; one run each). The new kernels made Tilde's
  reading 2.2–2.7× faster ([run](runs/2026-11-19-iphone16-qwen3-kernels.md)).
- **LiteRT-LM's numbers repeat.** Two more LiteRT-only runs read at 600 and 608 tok/s (256 tokens) and 811 (1,024),
  within 2% of the pass above. The 3,072-token prompt lost the GPU in all three runs that got that far, the same way
  each time, and one run was killed while loading at 430 MB of GPU buffers: loading a model that settles at 1.5 GB is
  close enough to the tab's ~1.9 GB that iOS sometimes kills it first ([runs](runs/2026-11-19-iphone16-qwen3-pass2.md#litert-lm-alone-twice-more-same-morning)).
- **Only Tilde finished every prompt size.** All three GPU runtimes that got past 256 tokens lost the GPU while
  reading a longer prompt: stock WebLLM at 1,024, paced WebLLM and LiteRT-LM at 3,072 (LiteRT-LM: "timeout was reached
  while reading back data"). That's WebKit's command-buffer limit, which Tilde sizes its work around.
- **Tilde now uses the least memory:** 865–873 MB with the f16 cache, against WebLLM's 1,125 and LiteRT-LM's 1,496
  (all counted by watching `createBuffer`).
- **Tilde slows down with context:** writing goes 26.9 → 21.7 → 14.6 tok/s from 256 to 3,072 tokens. Its attention
  kernel walks the cache one position at a time per head; splitting that work across workgroups is the next speed job.
  Prompt reading also slows with context (322 → 178 tok/s); the fused attention kernel only reaches ~120 GFLOP/s
  against the matmul's ~500.

## Earlier results, by runtime

| Runtime | Result | Speed | Memory | How it ended |
|---|---|---|---|---|
| **Tilde** (Nov 16) | ✅ runs | prompt **51 tok/s**, writing **12.6 tok/s**, reload from cache ~5 s | **1,123 MB** counted at a 4K context | Top 5 tokens match llama.cpp and 12/12 greedy tokens are identical. ([design.md](design.md#measured-on-an-iphone-16-safari-27-nov-16-2026)) |
| **WebLLM**, our own MLC 4-bit build (Nov 10) | ✅ with a patch | writing **13.7 tok/s**, first word in 15 s | – | A 3K-token prompt killed the tab while answering. ([Squiggle CLAUDE.md](https://github.com/KealJones/squiggle/blob/main/CLAUDE.md), [b9b9f87](https://github.com/KealJones/squiggle/commit/b9b9f87)) |
| **WebLLM**, QAD weights, one-sequence library (Nov 15) | ✅ up to 6K, with a patch | prompt ~100 tok/s unpaced (256-token pieces, ~2.5 s each) | 4K: 1,021 MB GPU, 1,112 peak. 6K: 1,065 MB, 1,200 peak | **8K killed the tab while loading.** ([a44fe37](https://github.com/KealJones/squiggle/commit/a44fe37)) |
| **wllama 3.13** (llama.cpp in WebAssembly) | ❌ no tokens | – | – | On its own it picks a build iOS 27.2 can't compile ("Memory64 is not enabled"). With its Safari build forced, the run got the tab killed, and Safari's auto-reload turned that into a crash loop. ([8115f27](https://github.com/KealJones/tilde/commit/8115f27), [b64a483](https://github.com/KealJones/tilde/commit/b64a483), [vs-wllama.html](../bench/vs-wllama.html)) |
| **wllama 3.8.1** | ❌ | – | – | Its llama.cpp predates LFM3. ([a694a1f](https://github.com/KealJones/tilde/commit/a694a1f)) |
| **wllama on the CPU** (memtest) | ⚠ short prompts only | 1.7 tok/s | 736 MB left on a short prompt | A full 3K-token prompt killed the tab while loading. ([LFM3 demo](https://github.com/KealJones/lm-experiments/blob/main/site/demos/lfm3-webllm/demo.js)) |
| **transformers.js**, ONNX on WebGPU (1.3 GB export) | ❌ never finished downloading | – | – | Three of four tries killed the tab. It holds a download in memory about twice over while loading it. ([LFM3 demo](https://github.com/KealJones/lm-experiments/blob/main/site/demos/lfm3-webllm/demo.js)) |

## What WebLLM needed to survive iOS 27

WebLLM is fast, but on iOS 27 it only stays up with patches reaching into its internals. Tilde avoids each of these
by design.

- **GPU watchdog.** From iOS 27.2.1 on, WebLLM lost the GPU on a plain "Hi" (a ~1,000-token prompt). It queues its
  256-token prefill pieces without waiting, ~2.5 s each, and WebKit kills the GPU process when a command buffer
  waits more than [3 s](https://github.com/WebKit/WebKit/blob/main/Source/WebGPU/WebGPU/Queue.mm#L492-L494).
  Squiggle cuts the pieces to 128 tokens and syncs between them by reaching into WebLLM's pipeline
  ([61ad3b4](https://github.com/KealJones/squiggle/commit/61ad3b4)). Tilde sizes every submit ahead of time from
  per-batch-size measurements instead.
- **"Object has already been disposed."** After a GPU loss, WebLLM's TVM runtime disposes itself and every later
  call fails until the model is loaded again ([688aaae](https://github.com/KealJones/squiggle/commit/688aaae)).
- **Prompt reuse.** A same-buffer `copyBufferToBuffer` lost the GPU device. With a workaround in place, reuse kept 61%
  of prompt tokens (3,392 of 5,536) from being read again, and the first word came at 7.2 s
  ([lab notebook](https://github.com/KealJones/squiggle/blob/main/docs/lab.md)). Tilde snapshots state at a safe cut
  and reads only the rest.
- **Compile jobs.** WebLLM had no LFM3 build at all. Making one took eight Hugging Face jobs and TVM built from source
  at an exact commit ([lfm3-mlc](https://github.com/KealJones/lm-experiments/blob/main/experiments/lfm3-mlc/README.md)).
  Tilde reads Liquid's official GGUF directly.

## Other phone-killers

- **Liquid's own LFM3 Omni 2B WebGPU demo** crashed the iPhone tab at ~80% loaded, because it loads every sense at
  once (~1.6 GB of weights). Tilde's Omni peaks at **1,280 MB** with vision swapped in and settles at **1,067 MB**
  ([4097517](https://github.com/KealJones/tilde/commit/4097517)).
- **Before LFM3:** Gemma 4 E2B and E4B on LiteRT got the tab killed, and Qwen3.5 2B on WebLLM loaded but died on its
  first reply ([lab notebook](https://github.com/KealJones/squiggle/blob/main/docs/lab.md)).
- **The 1.9 GB budget** comes from memcap: five rounds on the iPhone landed within about 0.1 GB of each other, and
  JavaScript, WebAssembly and the GPU all draw from the same budget
  ([memcap](https://github.com/KealJones/lm-experiments/blob/main/experiments/memcap/README.md)).

## Read these numbers fairly

- **Tilde doesn't use less memory than WebLLM.** At 4K on Nov 19 it held 1,065–1,092 MB against paced WebLLM's
  1,040–1,049 MB (1,123 against 1,021 on Nov 16), and the two are counted differently. Tilde's case is that it survives iOS 27 without patches, needs no compile jobs, runs official
  GGUFs directly and swaps senses in and out.
- **Prompt reading flipped on Nov 19.** On Nov 18 WebLLM read faster, paced or not (75.5 against Tilde's 50 tok/s
  at 256 tokens, 66.7 against 34.9 at 1,024). Since the Nov 19 kernel rewrite, Tilde reads 1.45× faster than stock
  WebLLM at 256 and 1.57–1.75× faster than paced WebLLM at every size (see the Nov 19 runs above). WebLLM still loads
  faster.
- **Writing speed is about even** when both run on the same day (12.1–12.3 against 12.0 tok/s at 256); the 13.7
  from Nov 10 was before iOS 27.2.1.
- **The wllama run doesn't record whether it used the GPU.** `result.webgpu` is only saved when a run finishes, so
  it was lost with the tab.

## Next runs

- Rerun the two WebLLM deaths (both while loading) and repeat every cell three times, with a fixed GPU workload
  timed before each test, so heat and leftover memory stop deciding results.
- Tilde at 6K and 8K. If Tilde loads at 8K, that's a direct win over WebLLM, which died loading there on Nov 15.
- Prompt reading: the matmul rewrite (64×64 tiles) and fused attention made Tilde 2.2–2.7× faster
  ([run](runs/2026-11-19-iphone16-qwen3-kernels.md)); LiteRT-LM still reads 1.5–3× faster. Before the rewrite,
  bigger chunks gave only 6–10% and a 4 s budget lost the GPU ([run](runs/2026-11-19-iphone16-qwen3-chunks.md));
  after it, 256-token chunks give 19–27%.
- Faster attention over long contexts: the fused kernel runs 64 workgroups per layer at 64 tokens; splitting each
  head's keys across workgroups should lift it past ~120 GFLOP/s. Decode attention needs the same.
- Run LFM3 1.7B and the other runtimes again against the new kernels.
