# Tilde after the kernel rewrite, Qwen3 0.6B, iPhone 16, Safari 27.2, Nov 19, 2026 (afternoon)

Tilde at commit a2b1657: 64×64 matmul tiles (`gemmQ4_0x64`) and fused prompt attention (`attnFlashFor`) for chunks of
33 tokens or more. Default (64-token chunks, 1 s submit budget) and "big chunks" (256 tokens, 2 s), from
[bench/compare.html](../../bench/compare.html). The same morning's numbers before the change are in
[2026-11-19-iphone16-qwen3-chunks.md](2026-11-19-iphone16-qwen3-chunks.md).

GPU apple (shader-f16); context 4096; 64 tokens written, greedy; one warm-up request each. Tilde: this page's copy.

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde | 256 | ran | 2.05 | 322 | 26.7 | 0.81 | 865 | 25 | Tilde counted 865 MB, longest submit 192 ms, chunk 64 at the end |
| Tilde, big chunks | 256 | ran | 1.91 | 407 | 27.1 | 0.64 | 879 | 26 | 256-token chunks, 2 s budget; Tilde counted 879 MB, longest submit 113 ms, chunk 256 at the end |
| Tilde | 1024 | ran | 2.63 | 267.5 | 21 | 3.84 | 865 | 39 | Tilde counted 865 MB, longest submit 284 ms, chunk 64 at the end |
| Tilde, big chunks | 1024 | ran | 1.96 | 336.8 | 21.5 | 3.05 | 879 | 35 | 256-token chunks, 2 s budget; Tilde counted 879 MB, longest submit 917 ms, chunk 256 at the end |
| Tilde | 3072 | ran | 2.04 | 177.8 | 14.5 | 17.29 | 865 | 109 | Tilde counted 865 MB, longest submit 534 ms, chunk 64 at the end |
| Tilde, big chunks | 3072 | ran | 2.09 | 213.5 | 14.4 | 14.4 | 879 | 26 | 256-token chunks, 2 s budget; Tilde counted 879 MB, longest submit 1823 ms, chunk 256 at the end |

Prompt reading against the morning (read tok/s, before → after):

| | 256 | 1,024 | 3,072 |
|---|---|---|---|
| Tilde | 135.9 → 322 (2.4×) | 116.8 → 267.5 (2.3×) | 79.8 → 177.8 (2.2×) |
| Tilde, big chunks | 149.4 → 407 (2.7×) | 127 → 336.8 (2.7×) | 84.5 → 213.5 (2.5×) |

Writing is unchanged (decode uses the same kernels). Memory dropped 8 MB (default) and 56 MB (big chunks) because the
attention score buffer is now only sized for chunks under 33 tokens.

Kernel timings from bench/kernel-bench.html on the same phone (per layer, 64-token chunk): Qwen3 matmuls 14.8 → 5.7 ms,
LFM3 28.8 → 10.05 ms; Qwen3 attention 2.85 → 2.65 ms at 256 positions, 8.35 → 4.45 at 1,024, 24.55 → 13.7 at 3,072.
