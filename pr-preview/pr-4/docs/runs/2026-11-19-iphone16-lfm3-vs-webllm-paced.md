# Tilde and paced WebLLM taking turns, LFM3 1.7B, iPhone 16, Safari 27.2, Nov 19, 2026 (afternoon, normal power)

From [bench/compare.html](../../bench/compare.html), Low Power Mode off, right after the stock-WebLLM run in
[2026-11-19-iphone16-lfm3-vs-webllm.md](2026-11-19-iphone16-lfm3-vs-webllm.md). Tilde at commit a2b1657 with
256-token chunks and a 2 s budget; WebLLM 0.2.87 with our LFM3 MLC build and Squiggle's `paceGpu` fix (128-token
prompt pieces, a sync and a 16 ms gap after each). Tests take turns by prompt size.

GPU apple (shader-f16); context 4096; 64 tokens written, greedy; one warm-up request each. Tilde: this page's copy.

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde, big chunks | 256 | ran | 5.65 | 155.4 | 12.3 | 1.66 | 1,092 | 23 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 346 ms, chunk 256 at the end |
| WebLLM + Squiggle's pacing | 256 | ran | 4.04 | 99 | 12.5 | 2.59 | 1,040 | 154 | LFM3-1.7B-QAD-q4f16_1-MLC |
| Tilde, big chunks | 1024 | ran | 5.61 | 156.3 | 12 | 6.56 | 1,092 | 38 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1557 ms, chunk 256 at the end |
| WebLLM + Squiggle's pacing | 1024 | ran | 3.85 | 95.7 | 11.8 | 10.7 | 1,041 | 70 | LFM3-1.7B-QAD-q4f16_1-MLC |
| Tilde, big chunks | 3072 | ran | 5.71 | 141 | 10.7 | 21.79 | 1,092 | 42 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1736 ms, chunk 256 at the end |
| WebLLM + Squiggle's pacing | 3072 | ran | 4.21 | 80.4 | 10.2 | 38.24 | 1,049 | 77 | LFM3-1.7B-QAD-q4f16_1-MLC |

| Prompt | Reading | Writing | First token |
|---:|---|---|---|
| 256 | Tilde 1.57× faster (155 / 99) | even (12.3 / 12.5) | 1.7 s / 2.6 s |
| 1,024 | Tilde 1.63× faster (156 / 96) | even (12.0 / 11.8) | 6.6 s / 10.7 s |
| 3,072 | Tilde 1.75× faster (141 / 80) | Tilde 5% faster (10.7 / 10.2) | 21.8 s / 38.2 s |

Both finished every size. WebLLM loads faster (about 4 s against 5.6 s) and uses ~50 MB less GPU memory (1,040–1,049
against 1,092 MB, Tilde's 256-token chunks costing ~27 MB of that). Its worst frame was 154 ms at 256 tokens.
