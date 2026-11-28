# WebLLM next to Tilde's new kernels, LFM3 1.7B, iPhone 16, Safari 27.2, Nov 19, 2026 (afternoon)

**The phone was on Low Power Mode for this run**, which throttles the GPU, so its WebLLM numbers aren't comparable
with the Tilde runs. The run with Low Power Mode off is [2026-11-19-iphone16-lfm3-vs-webllm.md](2026-11-19-iphone16-lfm3-vs-webllm.md).

Run right after the Tilde rerun in [2026-11-19-iphone16-lfm3-kernels.md](2026-11-19-iphone16-lfm3-kernels.md), from
[bench/compare.html](../../bench/compare.html). WebLLM runs our own LFM3 MLC build
(`kealjones/LFM3-1.7B-QAD-q4f16_1-MLC`) on unmodified WebLLM 0.2.87; "Squiggle's pacing" adds Squiggle's `paceGpu`
fix (128-token prompt pieces, a sync and a 16 ms gap after each).

GPU apple (shader-f16); context 4096; 64 tokens written, greedy; one warm-up request each.

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| WebLLM 0.2.87 | 256 | lost the GPU | – | – | – | – | 1,040 | – | reading: Object has already been disposed; GPU lost (unknown) during reading; LFM3-1.7B-QAD-q4f16_1-MLC |
| WebLLM + Squiggle's pacing | 256 | ran | 5.52 | 60.1 | 8.2 | 4.26 | 1,040 | 43 | LFM3-1.7B-QAD-q4f16_1-MLC |
| WebLLM 0.2.87 | 1024 | skipped | – | – | – | – | – | – | error at 256 tokens |
| WebLLM + Squiggle's pacing | 1024 | ran | 5.13 | 57.1 | 8.3 | 17.93 | 1,041 | 53 | LFM3-1.7B-QAD-q4f16_1-MLC |
| WebLLM 0.2.87 | 3072 | skipped | – | – | – | – | – | – | error at 256 tokens |
| WebLLM + Squiggle's pacing | 3072 | ran | 8.45 | 54.2 | 7.7 | 56.67 | 1,049 | 82 | LFM3-1.7B-QAD-q4f16_1-MLC |

Stock WebLLM alone again, a few minutes later: lost the GPU at 256 tokens the same way ("Object has already been
disposed"; GPU lost while reading); 1,024 and 3,072 skipped.

Against Tilde from the same half hour (default settings, read / write tok/s, first token); not a fair comparison,
see the note at the top:

| Prompt | Tilde | WebLLM + pacing | Tilde ahead by |
|---:|---|---|---|
| 256 | 138.2 / 12.3, 1.9 s | 60.1 / 8.2, 4.3 s | 2.3× reading, 1.5× writing |
| 1,024 | 137 / 11.9, 7.5 s | 57.1 / 8.3, 17.9 s | 2.4× reading, 1.4× writing |
| 3,072 | 119.4 / 10.4, 25.7 s | 54.2 / 7.7, 56.7 s | 2.2× reading, 1.35× writing |

Memory: Tilde 1,065 MB, WebLLM 1,040–1,049 MB.

Paced WebLLM was slower than on Nov 18 (75.5 / 12.1 at 256 then) but survived 3,072 this time (killed while loading
on Nov 18). Stock WebLLM survived 256 tokens on Nov 18 and lost the GPU there in both runs today.
