# Experiment 013: multi-seed refiner benchmark

## Question

Does the corrected Experiment 005 competitive recurrent refiner beat
farthest-point sampling across independently trained decoder/refiner seeds, or
was the earlier result specific to one learned decoder basin?

This is the first paper-quality benchmark slice, not a complete compression
benchmark. It resolves the predeclared primary operating point from Work
Package A of the [month-one paper-viability sprint](month-1-paper-viability-sprint.md).
The 8- and 10-bit rate points, external data, analytic surface metrics, and
actual bitstream comparisons remain open work.

## Protocol

- Procedural dataset: seven balanced families (`plane`, `corner`, `box`,
  `cylinder`, `sphere`, `beam`, and `pair`).
- Fixed data seed: 7, with 448 training, 140 validation, and 140
  parameter-OOD clouds. Dataset contents and split manifests are SHA-256
  hashed in the machine-readable result.
- Model seeds: 7, 17, and 29. Each seed independently trains its decoder and
  refiner; the three frozen decoder state hashes are distinct.
- Primary point: `N=256`, `K=16`, and 12 bits per coordinate, or 576 nominal
  coordinate payload bits. This is not an actual coded-rate claim.
- Matched arms: FPS, recurrent refiner without decoder-gradient feedback, and
  recurrent refiner with decoder-gradient feedback. Both trained refiner arms
  start from bitwise-identical weights and use identical data order, random
  seed, quantizer noise stream, and frozen decoder within each model seed.
- Encoder-side gradient feedback sees only the encoder input. Reconstruction
  is scored against the held-out target. At the primary `N=256` point the
  source is the complete input cloud.
- Outputs: exact-quantized free coordinates and a separately labeled post-hoc
  greedy unique projection to input points.
- Primary metric: symmetric squared-Chamfer RMSE. Relative improvements use
  aggregate RMSE over the paired seed/cloud matrix.
- Uncertainty: 10,000-replicate paired hierarchical bootstrap resampling model
  seeds and held-out clouds, reported as 95% intervals.

The implementation also evaluates all 12 combinations in
`N in {64, 128, 256}` and `K in {4, 8, 16, 32}` for each arm. The statistical
gate below is intentionally limited to the predeclared primary point.

## Primary result

| Split and method | Aggregate RMSE | vs FPS | Paired 95% CI |
|---|---:|---:|---:|
| Validation FPS | 0.18926 | 0.00% | -- |
| Validation no-feedback, free | 0.18119 | +4.26% | [-0.32%, 8.68%] |
| Validation input-gradient, free | **0.14510** | **+23.33%** | **[20.44%, 26.62%]** |
| Validation input-gradient, projected | **0.16179** | **+14.51%** | **[12.72%, 16.49%]** |
| Parameter-OOD FPS | 0.17157 | 0.00% | -- |
| Parameter-OOD no-feedback, free | 0.16309 | +4.94% | [0.34%, 9.40%] |
| Parameter-OOD input-gradient, free | **0.12846** | **+25.13%** | **[20.73%, 29.50%]** |
| Parameter-OOD input-gradient, projected | **0.14553** | **+15.18%** | **[13.01%, 17.51%]** |

The free input-gradient refiner beat FPS on all six seed/split comparisons:

| Model seed | Validation | Parameter OOD |
|---:|---:|---:|
| 7 | +20.84% | +20.79% |
| 17 | +22.35% | +25.74% |
| 29 | +26.28% | +28.08% |

The lower tail also remained positive. For the free input-gradient method, the
10th percentile of paired per-cloud improvement was 12.61% on validation and
12.81% on parameter OOD. Median paired improvements were 21.55% and 23.13%,
respectively. The method beat FPS in the aggregate for every primitive family;
validation gains ranged from 15.45% on spheres to 33.19% on planes, and OOD
gains ranged from 14.46% on spheres to 37.74% on paired structures.

## Mechanism and convergence

Decoder-gradient feedback improved over the matched no-feedback free refiner
by 19.92% on validation (95% CI 18.28-22.54) and 21.24% on parameter OOD
(18.46-24.71). This makes feedback more than a minor training detail in the
current architecture.

At validation, the aggregate free-coordinate RMSE changed with recurrent step
as follows:

| Step | Input-gradient | No feedback |
|---:|---:|---:|
| 0 (FPS) | 0.18926 | 0.18926 |
| 1 | 0.16743 | 0.18189 |
| 2 | 0.15863 | 0.18008 |
| 4 | 0.15090 | 0.18075 |
| 8 | **0.14510** | 0.18119 |

The legal input-gradient trajectory improves monotonically in aggregate and
continues to gain after the early steps. The no-feedback trajectory obtains a
small early improvement and then plateaus or slightly regresses. Strict
projection preserves a substantial gain but trails free coordinates, so both
the allocation mechanism and limited off-sample movement appear useful.

## Gate and interpretation

Gate A at the primary 12-bit point **passes**:

1. the legal input-gradient refiner beats FPS for every model seed on both
   validation and parameter OOD;
2. both pooled paired confidence intervals exclude zero;
3. the no-feedback arm improves at least one split; and
4. decoder independence and within-seed frozen/matched controls are verified by
   state, initialization, and data-order hashes.

This resolves the narrow uncertainty that the Experiment 005 result was a
one-decoder accident. It supports semi-amortized decoder-aware constellation
inference as the leading mechanism on the procedural distribution. It does not
yet establish general point-cloud compression, continuous-surface semantics,
or superiority at actual coded rate.

The strict-projection result is especially useful: it shows that the gain is
not exclusively a high-precision private coordinate code. However, projection
is post-hoc rather than a learned strict-subset encoder, and the free result may
still exploit decoder-specific off-sample locations. Work Package B must test
fresh independent surface samples, analytic surface distance, point-to-plane
error, and perturbation robustness before making a geometric representation
claim.

## Reproduce

```bash
.venv-train/bin/python -m pointconstellation.refiner_benchmark \
  --config configs/experiment_013_refiner_multiseed.json \
  --device mps
```

The first complete run used Python 3.13.14, PyTorch 2.13.0, and Apple MPS on
macOS arm64. It took 1,071.50 seconds. Checkpoints, per-cloud records, full-grid
curves, hashes, manifests, and `benchmark_metrics.json` are written under the
ignored `artifacts/local/experiment_013_refiner_multiseed/` directory. Finished
seed arms are reused on subsequent summary runs unless `--force` is supplied.
