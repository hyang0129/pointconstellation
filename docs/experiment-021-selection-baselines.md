# Experiment 021: non-learned selection baselines at identical bytes

Status: complete; the predeclared gate fails.

## Question and hypothesis

Experiment 020 compared the stabilized competitive refiner only with FPS.
Experiment 021 tests whether that result survives a broader set of deterministic,
source-only geometric baselines through the same six frozen Experiment 019
decoders.

The predeclared hypothesis is that the free-coordinate recurrent refiner has
lower official D1 and D2 RMSE than every non-learned arm on validation. The
primary statistic is paired relative RMSE improvement, computed from official
symmetric MSE and aggregated over all decoder seeds and validation clouds.

## Fixed protocol

- Data, source sampling, normals used for official evaluation, frozen decoder
  checkpoints, refiner checkpoints, and decoder/refiner seeds are inherited
  unchanged from Experiments 019 and 020.
- Every transmitted message has `K=8`, `q=12`, and exactly 50 serialized bytes:
  a 14-byte header plus a 36-byte fixed-width coordinate payload.
- Every method sees only the encoder-visible source cloud. Methods that rank
  multiple candidates may call the assigned frozen decoder and compare its
  reconstructions with that same source cloud using Chamfer loss.
- Decoder hashes are checked before and after evaluation. Coordinates must pass
  exact lattice and bitstream round-trip checks.
- Encode time is recorded per cloud and includes candidate generation,
  source-only decoder scoring, quantization, and serialization. Decode and
  official-metric time remain separate.
- Best-of-1 and best-of-N variants share a seed family, so the one-candidate arm
  is the first member of the corresponding larger candidate pool.

## Arms

| Method | Representation class | Fixed definition |
|---|---|---|
| `fps` | strict-subset | Existing mean-distant-start FPS control |
| `kmeans` | free-coordinate | k-means++ initialization and at most 50 Lloyd iterations |
| `kmeans_weighted` | free-coordinate | Same k-means procedure with weights proportional to the cubed 8-NN radius, an inverse local-density proxy |
| `poisson_disk` | strict-subset | Seeded greedy order; radius bisection followed by the first exactly eight feasible points |
| `fps_random_start` | strict-subset | One seeded random FPS start |
| `fps_random_start_best_of_8` | strict-subset | Eight seeded FPS starts, selected by source-only decoder Chamfer |
| `random_best_of_1` | strict-subset | One seeded random subset |
| `random_best_of_16` | strict-subset | Sixteen seeded random subsets, selected by source-only decoder Chamfer |
| `refiner` | free-coordinate | Each of the three frozen Experiment 019 recurrent refiners |

The strict-subset label describes selection before exact lattice quantization.
No point weights, cluster assignments, candidate scores, start indices, or point
order are serialized. K-means centroids are free coordinates and are not
presented as subsets.

## Analysis and gate

For each selection arm, split, and metric, the evaluator reports two paired
comparisons: the arm against FPS and the refiner factorial against that arm.
The hierarchical bootstrap resamples categories and clouds, paired decoder
seeds, and the common refiner-seed factor. A non-learned arm has one fixed
replicate per decoder/cloud cell.

The headline gate passes only if the refiner beats every selection arm on both
validation D1 and validation D2, with every 95% confidence interval for relative
RMSE improvement excluding zero. If any interval includes or falls below zero,
Table 1 must not claim a general refiner advantage: the headline changes to the
winning or statistically unresolved method-specific comparison. OOD results are
reported as transfer evidence and do not select the headline.

## Results

The complete run evaluated 10560 per-cloud rows (128 validation and 32 category-OOD clouds, six Experiment 019 decoders, eight selection methods, three refiner seeds) in 15.0 minutes on Apple MPS. Every stream was exactly 50 bytes; all contract checks passed.

**The predeclared gate fails** (`selection_baseline_gate_passes: false`).

### Absolute official RMSE at 50 bytes (12-bit grid units) and encode time

| Method | Class | Decoder evals | Val D1 | Val D2 | OOD D1 | OOD D2 | Encode ms |
|---|---|---:|---:|---:|---:|---:|---:|
| FPS (deterministic) | subset | 0 | 316.7 | 260.9 | 359.9 | 315.9 | 7.1 |
| FPS, random start | subset | 0 | 319.6 | 262.8 | 377.3 | 331.5 | 13.3 |
| k-means (free) | free | 0 | 322.3 | 272.6 | 344.5 | 300.1 | 9.9 |
| weighted k-means (free) | free | 0 | 328.3 | 278.9 | 364.8 | 312.6 | 23.9 |
| Poisson-disk | subset | 0 | 318.5 | 263.5 | 363.3 | 318.4 | 248.3 |
| random subset | subset | 0 | 342.2 | 264.8 | 367.1 | 303.0 | 10.5 |
| FPS random start, best of 8 by decoder | subset | 8 | 284.4 | 209.3 | 322.6 | 263.1 | 62.0 |
| random subset, best of 16 by decoder | subset | 16 | 272.4 | 160.8 | 286.3 | 192.0 | 97.1 |
| recurrent refiner (3 seeds) | free | 8 steps | 264.4 | 170.5 | 292.3 | 209.2 | 21.6 |

### Paired bootstrap: refiner versus each selection arm

Relative RMSE improvement of the refiner over the arm (positive favours the refiner); 95% paired hierarchical bootstrap CI; decoders in which the refiner wins.

| Arm | Val D1 | Val D2 | OOD D1 | OOD D2 |
|---|---|---|---|---|
| FPS (deterministic) | +16.52% [11.43, 21.79] 6/6 | +34.65% [27.05, 40.82] 6/6 | +18.80% [8.77, 25.50] 6/6 | +33.78% [26.62, 41.55] 6/6 |
| FPS, random start | +17.26% [12.06, 22.42] 6/6 | +35.13% [26.50, 41.92] 6/6 | +22.54% [12.21, 29.15] 6/6 | +36.89% [28.60, 43.23] 6/6 |
| k-means (free) | +17.95% [12.89, 23.26] 6/6 | +37.45% [28.34, 45.57] 6/6 | +15.16% [7.75, 22.03] 6/6 | +30.29% [19.25, 40.17] 6/6 |
| weighted k-means (free) | +19.46% [14.64, 24.42] 6/6 | +38.86% [29.96, 46.56] 6/6 | +19.89% [9.10, 30.01] 6/6 | +33.09% [22.68, 42.75] 6/6 |
| Poisson-disk | +16.97% [11.84, 22.08] 6/6 | +35.29% [26.91, 42.26] 6/6 | +19.56% [7.57, 28.50] 6/6 | +34.30% [24.10, 43.29] 6/6 |
| random subset | +22.73% [17.11, 28.82] 6/6 | +35.62% [25.06, 45.08] 6/6 | +20.38% [12.73, 27.72] 6/6 | +30.97% [19.78, 43.95] 6/6 |
| FPS random start, best of 8 by decoder | +7.03% [4.12, 10.12] 6/6 | +18.54% [10.82, 25.59] 6/6 | +9.41% [1.89, 17.32] 6/6 | +20.49% [9.22, 30.02] 6/6 |
| random subset, best of 16 by decoder | +2.94% [1.23, 4.53] 6/6 | -6.05% [-13.51, 0.37] 1/6 | -2.10% [-11.38, 5.09] 2/6 | -8.95% [-25.17, 9.27] 2/6 |

### Selection arms versus deterministic FPS

| Arm | Val D1 | Val D2 | OOD D1 | OOD D2 |
|---|---|---|---|---|
| FPS, random start | -0.90% [-3.94, 2.30] 2/6 | -0.73% [-5.50, 4.22] 3/6 | -4.83% [-10.89, 1.51] 0/6 | -4.94% [-14.42, 5.99] 0/6 |
| k-means (free) | -1.75% [-8.01, 3.79] 2/6 | -4.47% [-15.76, 4.50] 2/6 | +4.28% [-6.09, 13.18] 3/6 | +5.00% [-10.94, 17.14] 3/6 |
| weighted k-means (free) | -3.66% [-9.18, 1.39] 0/6 | -6.88% [-17.12, 1.46] 0/6 | -1.37% [-12.35, 7.51] 2/6 | +1.03% [-15.25, 14.00] 3/6 |
| Poisson-disk | -0.55% [-4.69, 3.81] 1/6 | -0.99% [-8.09, 5.94] 2/6 | -0.95% [-9.70, 6.66] 3/6 | -0.79% [-11.60, 9.61] 3/6 |
| random subset | -8.04% [-15.65, -0.98] 0/6 | -1.50% [-14.23, 9.43] 2/6 | -1.98% [-19.79, 11.00] 2/6 | +4.07% [-19.45, 18.75] 3/6 |
| FPS random start, best of 8 by decoder | +10.20% [6.46, 14.72] 6/6 | +19.78% [14.55, 25.12] 6/6 | +10.36% [3.99, 16.03] 6/6 | +16.71% [8.85, 24.49] 6/6 |
| random subset, best of 16 by decoder | +13.99% [8.38, 19.91] 6/6 | +38.38% [30.85, 45.07] 6/6 | +20.47% [5.82, 31.60] 6/6 | +39.22% [26.19, 47.23] 6/6 |

### Interpretation

1. **One-shot geometric selection does not beat FPS.** k-means, weighted k-means, Poisson-disk, random-start FPS, and a single random subset are all statistically indistinguishable from deterministic FPS (every CI includes zero). The refiner's 16--20% D1 and 30--39% D2 margins over FPS therefore hold against every one-shot arm as well.
2. **Decoder-in-the-loop random search nearly closes the gap.** Choosing the best of 16 random strict subsets by source-only decoder Chamfer (16 decoder evaluations, no training, no gradients) reaches validation D1 272.4 versus the refiner's 264.4: the refiner's remaining D1 advantage is +2.94% [1.23, 4.53] on validation and −2.10% [−11.38, 5.09] on category OOD. On D2 the random best-of-16 arm is *better* than the refiner: −6.05% [−13.51, 0.37] validation and −8.95% [−25.17, 9.27] OOD. Best-of-8 random-start FPS recovers roughly half the refiner gain.
3. **The gate fails**, and with it the claim that the learned recurrent refiner is what makes decoder-aware constellations good. Consistent with the Experiment 019 Adam-STE probe, the operative mechanism is *any* source-only search that consults the frozen decoder; the learned amortization contributes at most a few percent of D1 at 50 bytes and is not on the D2 Pareto front against 16 random trials. The refiner's remaining case is cost: ~22 ms per cloud for 8 recurrent steps versus ~97 ms for 16 random decoder evaluations on MPS.
4. **Consequence for the paper.** The headline must be reframed as *decoder-aware constellation selection beats geometric selection* (a result about the representation and the frozen decoder), with the refiner presented as an amortized approximation whose value is measured on an encode-time versus distortion curve (Experiment 022). Claims that the recurrent or competitive mechanism is itself the contribution are not supported at this rate.

Numbers are read from `artifacts/local/experiment_021_selection_baselines/official_metrics.json` and `official_per_cloud.jsonl`; run manifest and tool hashes are in `run_manifest.json`.

## Reproduction

Run a four-cloud-per-split smoke on macOS with:

```bash
.venv-train/bin/python -m pointconstellation.official_stability \
  --config configs/experiment_021_selection_baselines.json \
  --device mps \
  --max-clouds-per-split 4
```

Remove `--max-clouds-per-split 4` for the full predeclared evaluation. The output
is resumable only when the run manifest, dataset identity, tool hash, method
list, and all checkpoint hashes remain unchanged.
