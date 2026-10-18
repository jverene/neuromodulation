# P1/P1b analysis (mechanical, preregistered rules)

g_aligned_own = H_aligned(K=0, own-grown) − H_aligned(K=3 m=0, own-grown); positive =
the K=3 parent of the pair is more robust under the corrected protocol.
g_plain_shared = the same gap under the ORIGINAL protocol (plain, shared state).

| pair | K=3 m=0 own (plain/aligned, surv) | K=0 shared (plain/aligned, surv) | K=0 own (plain/aligned, surv) | g_aligned_own | g_plain_shared |
|---|---|---|---|---|---|
| p1_s0 | 0.026/0.010 (1.00) | 0.071/0.054 (0.88) | 0.025/0.018 (1.00) | +0.0083 | +0.0450 |
| p1_s1 | 0.030/0.016 (1.00) | 0.046/0.046 (1.00) | 0.046/0.046 (1.00) | +0.0297 | +0.0161 |
| p1_s2 | 0.026/0.009 (1.00) | 0.125/0.107 (0.17) | 0.045/0.032 (0.95) | +0.0230 | +0.0988 |
| p1_s3 | 0.021/0.010 (1.00) | 0.051/0.031 (1.00) | 0.034/0.022 (1.00) | +0.0121 | +0.0309 |
| p1_s4 | 0.035/0.017 (1.00) | 0.062/0.044 (1.00) | 0.030/0.020 (1.00) | +0.0027 | +0.0264 |
| p1b_t10 | 0.028/0.014 (1.00) | 0.059/0.026 (0.97) | 0.029/0.012 (1.00) | -0.0016 | +0.0308 |
| p1b_t11 | 0.030/0.012 (1.00) | 0.068/0.040 (0.97) | 0.033/0.028 (1.00) | +0.0154 | +0.0373 |
| p1b_t12 | 0.035/0.019 (0.90) | 0.114/0.098 (0.33) | 0.022/0.014 (1.00) | -0.0047 | +0.0794 |
| p1b_t13 | 0.029/0.010 (1.00) | 0.024/0.015 (1.00) | 0.020/0.012 (1.00) | +0.0018 | -0.0051 |

## Decision rules (fixed before runs)
- **P1** (channel-exercise pairs, n=5): g_aligned_own > 0 in 5/5; median +0.0121 → channel-experience effect **SUPPORTED**.
  - g_aligned_own per seed: [0.0083, 0.0297, 0.023, 0.0121, 0.0027]
  - original-protocol gaps (plain, shared): [0.045, 0.0161, 0.0988, 0.0309, 0.0264] (5/5 positive)
- **P1b** (init-luck null pairs, n=4): g_aligned_own per pair [-0.0016, 0.0154, -0.0047, 0.0018] (2/4 positive)
  - original-protocol gaps (plain, shared): [0.0308, 0.0373, 0.0794, -0.0051] (3/4 positive)
