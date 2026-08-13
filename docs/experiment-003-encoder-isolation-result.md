# Experiment 003: relation-decoder encoder isolation result

Run date: 2026-08-10

Experiment 003 held the relation-aware decoder controls constant while
comparing FPS, learned hard input-subset selection, and a 3 x 3 grid over soft
projection temperature and anchor surface-loss strength.

The experiment followed the predefined
[plan and gate](experiment-003-encoder-isolation-plan.md) without changing its
selection rule or thresholds after observing results.

## Configuration

- Apple M5 MPS, PyTorch 2.13.0
- 256 input/output points
- `K=16`, 12 bits per coordinate, 576-bit coordinate payload
- 448 training, 140 validation, and 140 parameter-OOD clouds
- 12 epochs, batch size 8, seed 7
- one FPS reference, one learned hard subset, and nine soft-projection runs
- 877.59 seconds total wall time

Command:

```bash
.venv-train/bin/python -m pointconstellation.encoder_isolation \
  --config configs/experiment_003_encoder_isolation.json \
  --device mps
```

## Aggregate result

| Condition | Projection temperature | Surface weight | Validation RMSE | Validation surface RMSE | Repulsion | OOD RMSE | OOD surface RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| FPS | - | 0.1 | **0.12665** | 0.00025 | 0.00000 | **0.11687** | 0.00025 |
| Hard subset | - | 0.1 | 0.16226 | **0.00026** | 0.00064 | 0.16112 | **0.00025** |
| Soft | 0.005 | 0.1 | 0.16736 | 0.02979 | 0.00937 | 0.17295 | 0.02177 |
| Soft | 0.005 | 1.0 | 0.19157 | 0.02438 | 0.00937 | 0.19111 | 0.02456 |
| Soft | 0.005 | 10.0 | 0.17318 | 0.03973 | 0.00937 | 0.17917 | 0.02299 |
| Soft | 0.020 | 0.1 | 0.19050 | 0.04738 | 0.00937 | 0.20064 | 0.04221 |
| Soft | 0.020 | 1.0 | 0.17713 | 0.04872 | 0.00937 | 0.18248 | 0.04217 |
| Soft | 0.020 | 10.0 | 0.17088 | 0.06606 | 0.00937 | 0.17267 | 0.04059 |
| Soft | 0.050 | 0.1 | 0.17456 | 0.07089 | 0.00937 | 0.17969 | 0.07149 |
| Soft | 0.050 | 1.0 | 0.18598 | 0.08161 | 0.00937 | 0.19756 | 0.05173 |
| Soft | 0.050 | 10.0 | 0.17556 | 0.08109 | 0.00937 | 0.17893 | 0.05446 |

Per-family metrics, parameter counts, and timing are retained in the generated
`encoder_isolation.json` artifact.

## Gate result: fail

The validation-only selection rule chose the hard-subset encoder because it had
the lowest validation distortion among candidates satisfying the `0.01`
surface-RMSE limit.

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Validation gap versus FPS | <= 5% | 28.12% worse | Fail |
| Parameter-OOD gap versus FPS | <= 5% | 37.86% worse | Fail |
| Validation surface RMSE | <= 0.01 | 0.00026 | Pass |
| Parameter-OOD surface RMSE | <= 0.01 | 0.00025 | Pass |

No soft-projection condition met the validation surface constraint. The best
soft distortion was 0.16736 at temperature 0.005 and surface weight 0.1, still
32.14% worse than FPS on validation.

## The important mechanism

The soft encoders collapsed their anchors. With `K=16` and the current
repulsion margin of 0.1, 16 coincident anchors produce:

```text
(15 / 16) * 0.1^2 = 0.009375
```

Every soft condition ended with repulsion approximately 0.009375, independently
of projection temperature and surface weight. This is effectively the maximum
off-diagonal repulsion penalty and is direct evidence that the learned queries
converged to nearly the same coordinate.

The surface loss is one-sided: it asks every anchor to be near some input point
but never asks the anchors to cover the input. Increasing its weight cannot
distinguish one on-surface cluster from a well-distributed constellation. The
configured repulsion contribution is also too small to offset reconstruction
optimization after collapse.

The hard-subset encoder fixes duplicate index selection and removes off-surface
coding, but uniqueness alone does not guarantee coverage. Its nonzero repulsion
and large FPS gap show that it still selects points too locally or unstably.

## Decision

Experiment 003 fails, so adaptive-`K` training remains blocked. Do not spend
another sweep on projection temperature or surface weight alone.

The next encoder objective should explicitly reward coverage, for example:

```text
L_coverage = mean over x in X of min over z in Z ||x - z||^2
```

This is the input-to-constellation half of Chamfer distance. Evaluate it with a
substantially stronger diversity term and the hard-subset encoder. A second
promising control is to start from FPS and learn bounded point swaps or residual
selection rather than learning an unconstrained set from symmetric queries.

The next test should keep the same K=16 FPS reference and predeclare whether a
coverage-trained hard selector can close the validation and OOD gaps while
remaining an actual quantized input subset.
