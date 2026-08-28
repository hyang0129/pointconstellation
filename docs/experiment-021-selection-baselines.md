# Experiment 021: non-learned selection baselines at identical bytes

Status: implemented; full ModelNet40 evaluation pending.

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

Pending. Populate this section from
`artifacts/local/experiment_021_selection_baselines/official_metrics.json`
after the complete run; do not promote smoke-run estimates.

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
