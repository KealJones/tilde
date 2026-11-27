# Tilde after the kernel rewrite, LFM3 1.7B, iPhone 16, Safari 27.2, Nov 19, 2026 (afternoon)

Tilde at commit a2b1657 (64×64 matmul tiles, fused prompt attention), default (64-token chunks, 1 s budget) and big
chunks (256 tokens, 2 s), from [bench/compare.html](../../bench/compare.html). The Nov 18 head-to-head is in
[2026-11-18-iphone16-lfm3.md](2026-11-18-iphone16-lfm3.md).

GPU apple (shader-f16); context 4096; 64 tokens written, greedy; one warm-up request each. Tilde: this page's copy.

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde | 256 | ran | 5.1 | 139.3 | 12.3 | 1.85 | 1,065 | 25 | Tilde counted 1065 MB, longest submit 491 ms, chunk 64 at the end |
| Tilde, big chunks | 256 | ran | 4.46 | 156.8 | 12.3 | 1.64 | 1,092 | 25 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 243 ms, chunk 256 at the end |
| Tilde | 1024 | tab killed | – | – | – | – | 865 | – | killed during loading at 865 MB of GPU buffers |
| Tilde, big chunks | 1024 | ran | 4.17 | 157 | 12 | 6.53 | 1,092 | 30 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1552 ms, chunk 256 at the end |
| Tilde | 3072 | skipped | – | – | – | – | – | – | died at 1024 tokens |
| Tilde, big chunks | 3072 | ran | 6.67 | 143.9 | 11.1 | 21.36 | 1,092 | 47 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 2004 ms, chunk 256 at the end |

Prompt reading against Nov 18 (read tok/s; Nov 18 Tilde had an f32 KV cache and 64-token chunks):

| | 256 | 1,024 | 3,072 |
|---|---|---|---|
| Tilde, Nov 18 | 50 | 34.9 | 40.8 |
| Tilde, default (rerun below) | 138.2–139.3 (2.8×) | 137 (3.9×) | 119.4 (2.9×) |
| Tilde, big chunks (both runs) | 156.6–156.8 (3.1×) | 152.2–157 (4.4–4.5×) | 124.8–143.9 (3.1–3.5×) |

- The default run at 1,024 was killed **while loading**, at 865 MB of GPU buffers, below the 1,065 MB the same
  configuration used at 256 tokens. Loading doesn't run the new kernels (it streams weights), and the bigger
  configuration loaded fine right after, so this looks like the pattern seen with WebLLM on Nov 18 and LiteRT-LM on
  Nov 19: iOS hadn't yet freed the previous page's memory. The rerun below finished every size.
- Reading slows little with context (157 → 125–144 tok/s from 1,024 to 3,072 with big chunks): only 6 of LFM3's 24 layers attend to the
  whole context; 5 use a sliding window and 13 are short convolutions.
- The 3,072-token big-chunk run's longest submit was 2.0 s, at its budget and about 1 s under WebKit's limit.

## Rerun, five minutes later

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde | 256 | ran | 4.33 | 138.2 | 12.3 | 1.86 | 1,065 | 28 | Tilde counted 1065 MB, longest submit 450 ms, chunk 64 at the end |
| Tilde, big chunks | 256 | ran | 4.34 | 156.6 | 12.3 | 1.65 | 1,092 | 23 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 239 ms, chunk 256 at the end |
| Tilde | 1024 | ran | 6.24 | 137 | 11.9 | 7.48 | 1,065 | 40 | Tilde counted 1065 MB, longest submit 488 ms, chunk 64 at the end |
| Tilde, big chunks | 1024 | ran | 6.29 | 152.2 | 11.6 | 6.74 | 1,092 | 24 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1678 ms, chunk 256 at the end |
| Tilde | 3072 | ran | 4.17 | 119.4 | 10.4 | 25.74 | 1,065 | 37 | Tilde counted 1065 MB, longest submit 588 ms, chunk 64 at the end |
| Tilde, big chunks | 3072 | ran | 4.39 | 124.8 | 10 | 24.62 | 1,092 | 26 | 256-token chunks, 2 s budget; Tilde counted 1092 MB, longest submit 1912 ms, chunk 256 at the end |

The default configuration finished every size, so the earlier kill was the load-time kind, not the kernels. The
256 and 1,024 cells repeat within 3%; the 3,072 big-chunk cell read 13% slower than the first run (144 → 125), and
writing was a little slower everywhere at 3,072, likely heat after back-to-back runs. On LFM3 big chunks gain 4–13%
over the default, less than on Qwen3 (19–27%).
