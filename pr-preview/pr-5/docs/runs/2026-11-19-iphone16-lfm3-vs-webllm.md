# Tilde and WebLLM taking turns, LFM3 1.7B, iPhone 16, Safari 27.2, Nov 19, 2026 (afternoon, normal power)

From [bench/compare.html](../../bench/compare.html), Low Power Mode off. The WebLLM run just before
([2026-11-19-iphone16-lfm3-webllm.md](2026-11-19-iphone16-lfm3-webllm.md)) was on Low Power Mode, so this one
replaces it for comparisons. Tilde at commit a2b1657 with 256-token chunks and a 2 s budget; WebLLM 0.2.87 stock with
our LFM3 MLC build. Tests take turns by prompt size.

GPU apple (shader-f16); context 4096; 64 tokens written, greedy; one warm-up request each. Tilde: this page's copy.

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde, big chunks | 256 | ran | 5.83 | 149.1 | 12.1 | 1.73 | 1,092 | 29 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 308 ms, chunk 256 at the end |
| WebLLM 0.2.87 | 256 | ran | 4.02 | 102.5 | 12.7 | 2.5 | 1,040 | 34 | LFM3-1.7B-QAD-q4f16_1-MLC |
| Tilde, big chunks | 1024 | ran | 5.63 | 149.2 | 11.4 | 6.87 | 1,092 | 35 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1634 ms, chunk 256 at the end |
| WebLLM 0.2.87 | 1024 | tab killed | – | – | – | – | – | – | killed during loading; LFM3-1.7B-QAD-q4f16_1-MLC |
| Tilde, big chunks | 3072 | ran | 5.64 | 115 | 8.9 | 26.73 | 1,092 | 33 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1884 ms, chunk 256 at the end |
| WebLLM 0.2.87 | 3072 | skipped | – | – | – | – | – | – | died at 1024 tokens |

- At 256 tokens Tilde read 1.45× faster (149 against 102.5 tok/s) and reached the first token sooner (1.7 s against
  2.5); WebLLM wrote 5% faster (12.7 against 12.1).
- WebLLM was killed while loading the 1,024 run, as on Nov 18, so there's no stock WebLLM number above 256 tokens.
- Tilde's 3,072 run was slower than the earlier two (115 against 125–144 reading, 8.9 against 10–11 writing).
  It ran last, after about 20 minutes of back-to-back tests.
