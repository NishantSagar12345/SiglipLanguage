Measured on **Tesla T4**, PyTorch 2.11.0+cu130, Transformers 5.20.2, image tower **float16** (logits FP32), 300 episodes (seed 42), run 2026-11-08T17:33:26Z.

Validation: oracle success 100% with optimal path length in 100% of episodes; pixel proposals equal the true object cells in 100.0% of first observations; 196-token crop scoring vs `SiglipModel(...).logits_per_image` (FP32, 40 crops × 6 instructions): max |Δ logit| = 1.24e-05 (|logit| ≤ 12.6), argmax agreement 100% — **passed**.
float16 vs FP32 first-step target choice agreement: 100.0% (8 episodes).

| Policy | Success | 95% CI | Δ vs 196 (pp) [paired 95% CI] | Target selection (step 0) | Δ sel. vs 196 (pp) | Steps (successes) | SPL |
|---|---:|---|---|---:|---:|---:|---:|
| SigLIP 196 tokens | 89.3% | [85.7, 92.7] | +0.0 [+0.0, +0.0] | 97.0% | +0.0 | 7.91 (opt. 7.91) | 0.893 |
| SigLIP 144 tokens | 88.7% | [85.0, 92.3] | -0.7 [-3.3, +2.0] | 97.0% | +0.0 | 7.80 (opt. 7.79) | 0.886 |
| SigLIP 100 tokens | 90.0% | [86.7, 93.3] | +0.7 [-3.0, +4.3] | 98.0% | +1.0 | 7.77 (opt. 7.77) | 0.900 |
| SigLIP 49 tokens | 80.0% | [75.3, 84.3] | -9.3 [-14.3, -4.7] | 89.7% | -7.3 | 7.89 (opt. 7.84) | 0.797 |
| SigLIP 196, shuffled instruction | 5.7% | [3.3, 8.3] | -83.7 [-88.0, -79.3] | 8.3% | -88.7 | 8.12 (opt. 7.76) | 0.055 |
| random candidate | 20.3% | [16.0, 25.0] | -69.0 [-74.7, -63.0] | 20.3% | -76.7 | 7.90 (opt. 7.90) | 0.203 |
| nearest candidate | 20.7% | [16.3, 25.3] | -68.7 [-74.0, -63.0] | 20.7% | -76.3 | 4.45 (opt. 4.45) | 0.207 |
| oracle (GT target) | 100.0% | [100.0, 100.0] | +10.7 [+7.3, +14.3] | 100.0% | +3.0 | 7.94 (opt. 7.94) | 1.000 |

**Episode outcomes (counts)**

| Policy | success | timeout | wrong_shape (colour right) | wrong_colour (shape right) | wrong_colour_and_shape |
|---|---:|---:|---:|---:|---:|
| SigLIP 196 tokens | 268 | 30 | 1 | 1 | 0 |
| SigLIP 144 tokens | 266 | 31 | 1 | 2 | 0 |
| SigLIP 100 tokens | 270 | 29 | 1 | 0 | 0 |
| SigLIP 49 tokens | 240 | 36 | 21 | 3 | 0 |
| SigLIP 196, shuffled instruction | 17 | 63 | 55 | 91 | 74 |
| random candidate | 61 | 0 | 70 | 94 | 75 |
| nearest candidate | 62 | 0 | 62 | 97 | 79 |
| oracle (GT target) | 300 | 0 | 0 | 0 | 0 |

**Latency per decision** (median / p90 ms over timed reps; GPU timings start from preprocessed crops already on the GPU, CPU stages are timed separately and do not depend on the token count)

| Stage | Tokens | Median (ms) | p90 (ms) | Speed-up vs 196 |
|---|---:|---:|---:|---:|
| `cpu_bfs_planning` | – | 0.09 | 0.13 | – |
| `cpu_crop_and_preprocess` | – | 3.65 | 3.89 | – |
| `cpu_render_and_proposals` | – | 6.98 | 7.20 | – |
| `decision_gpu_64_agents_batched` | 196 | 914.48 | 953.31 | 1.00× |
| `decision_gpu_64_agents_batched` | 144 | 665.97 | 693.11 | 1.37× |
| `decision_gpu_64_agents_batched` | 100 | 446.65 | 466.94 | 2.05× |
| `decision_gpu_64_agents_batched` | 49 | 236.43 | 246.31 | 3.87× |
| `decision_gpu_single_agent` | 196 | 19.94 | 20.89 | 1.00× |
| `decision_gpu_single_agent` | 144 | 14.06 | 14.64 | 1.42× |
| `decision_gpu_single_agent` | 100 | 11.95 | 12.36 | 1.67× |
| `decision_gpu_single_agent` | 49 | 8.29 | 8.70 | 2.41× |
| `encoder_only_single_agent` | 196 | 16.87 | 17.96 | 1.00× |
| `encoder_only_single_agent` | 144 | 11.87 | 12.74 | 1.42× |
| `encoder_only_single_agent` | 100 | 10.04 | 10.86 | 1.68× |
| `encoder_only_single_agent` | 49 | 6.95 | 7.30 | 2.43× |

Bootstrap: 10000 paired resamples of the episodes, seed 42, percentile intervals.
