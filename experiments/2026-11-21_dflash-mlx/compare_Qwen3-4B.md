# Matched drafter comparison on Qwen3-4B edits

235 edits present in every run (DFlash hybrid runs that diverged through bf16 numerics excluded).

## functions (n=147)

| Method | Tokens per target call | Accepted per draft on new code | H100 projection (optimistic / mid / pessimistic drafter cost) |
|---|---:|---:|---|
| Copy only | 6.4 | – | 5.0× |
| DFlash alone | 8.9 | 3.52 | 5.6× |
| Copy + DFlash | 13.1 | 3.44 | 8.6× |
| Qwen3-0.6B alone, γ=7 | 6.7 | 3.44 | 2.9× / 2.0× / 1.5× |
| Copy + Qwen3-0.6B, γ=7 | 12.9 | 3.34 | 6.6× / 4.9× / 3.9× |
| Copy + Qwen3-0.6B, γ=15 | 14.9 | 4.98 | 5.3× / 3.4× / 2.5× |
| Qwen3-1.7B alone, γ=7 | 7.0 | 3.85 | 1.6× / 1.5× / 1.3× |
| Copy + Qwen3-1.7B, γ=7 | 13.7 | 3.86 | 4.4× / 4.0× / 3.6× |
| Copy + Qwen3-1.7B, γ=15 | 16.6 | 6.12 | 3.1× / 2.7× / 2.4× |

## full files (n=88)

| Method | Tokens per target call | Accepted per draft on new code | H100 projection (optimistic / mid / pessimistic drafter cost) |
|---|---:|---:|---|
| Copy only | 6.2 | – | 4.9× |
| DFlash alone | 10.3 | 4.94 | 6.5× |
| Copy + DFlash | 14.8 | 4.82 | 9.8× |
| Qwen3-0.6B alone, γ=7 | 7.0 | 4.20 | 3.0× / 2.1× / 1.6× |
| Copy + Qwen3-0.6B, γ=7 | 14.0 | 4.14 | 7.1× / 5.3× / 4.2× |
| Copy + Qwen3-0.6B, γ=15 | 16.6 | 6.34 | 6.1× / 4.0× / 3.0× |
| Qwen3-1.7B alone, γ=7 | 7.3 | 4.84 | 1.7× / 1.5× / 1.3× |
| Copy + Qwen3-1.7B, γ=7 | 14.9 | 4.76 | 4.8× / 4.4× / 3.9× |
| Copy + Qwen3-1.7B, γ=15 | 18.5 | 7.77 | 3.6× / 3.2× / 2.8× |

DFlash rows use the fixed calibrated cycle cost, so their three projections coincide. r ranges: Qwen3-0.6B: (0.15, 0.3, 0.45), Qwen3-1.7B: (0.43, 0.5, 0.6).
