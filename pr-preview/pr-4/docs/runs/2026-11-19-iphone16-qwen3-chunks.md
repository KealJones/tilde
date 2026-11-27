# Tilde's chunk size and submit budget, Qwen3 0.6B, iPhone 16, Safari 27.2, Nov 19, 2026

Default Tilde (64-token chunks, 1 s submit budget) next to "Tilde, big chunks" (256-token chunks, 2 s budget), same
session, from [bench/compare.html](../../bench/compare.html).

| Runtime | Prompt | Outcome | Load s | Read tok/s | Write tok/s | First token s | GPU MB | Worst frame ms | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Tilde | 256 | ran | 1.78 | 135.9 | 27.5 | 1.89 | 873 | 37 | Tilde counted 873 MB, longest submit 477 ms, chunk 64 at the end |
| Tilde, big chunks | 256 | ran | 2.39 | 149.4 | 28.4 | 1.72 | 935 | 116 | 256-token chunks, 2 s budget; Tilde counted 935 MB, longest submit 219 ms, chunk 256 at the end |
| Tilde | 1024 | ran | 1.63 | 116.8 | 22 | 8.78 | 873 | 36 | Tilde counted 873 MB, longest submit 651 ms, chunk 64 at the end |
| Tilde, big chunks | 1024 | ran | 1.76 | 127 | 21.6 | 8.07 | 935 | 26 | 256-token chunks, 2 s budget; Tilde counted 935 MB, longest submit 2088 ms, chunk 256 at the end |
| Tilde | 3072 | ran | 1.68 | 79.8 | 14.5 | 38.52 | 873 | 42 | Tilde counted 873 MB, longest submit 1014 ms, chunk 64 at the end |
| Tilde, big chunks | 3072 | ran | 1.87 | 84.5 | 14.5 | 36.37 | 935 | 42 | 256-token chunks, 2 s budget; Tilde counted 935 MB, longest submit 2149 ms, chunk 256 at the end |

With a 4 s budget (`?bigBudget=4000`): 256 tokens read at 151 tok/s; the 1,024-token run logged an 8.3 s submit but
the page was hidden for part of it (Safari pauses GPU work while hidden, so that time isn't GPU time); the
3,072-token run lost the GPU ("destroyed") 27 s into reading, and Tilde then hung until the bench gave up, which is
what commit c9d2f5e fixes (a lost GPU now throws at once).
