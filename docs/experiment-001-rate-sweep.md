# Experiment 001: constellation rate and precision sweep

Run date: 2026-08-10

This sweep asks whether the current fixed-rate model converts a larger
coordinate message into lower distortion. It independently trains the learned
and FPS variants at 12 operating points.

## Configuration

- Apple M5 MPS, PyTorch 2.13.0
- 256 input/output points
- `K in {4, 8, 16, 32}`
- coordinate precision in `{8, 10, 12}` bits
- 224 training and 70 in-distribution validation clouds
- one seed and three epochs per model
- matched decoder architecture, initialization, data order, and quantizer for
  learned-versus-FPS comparisons at each point
- 38.55 seconds total wall time

Command:

```bash
.venv-train/bin/python -m pointconstellation.sweep \
  --config configs/experiment_001_rate_sweep.json
```

## Aggregate result

| `K` | Bits/coord | Payload bits | Bits/input point | Learned RMSE | FPS RMSE | Learned reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 96 | 0.375 | 0.36695 | 0.39011 | 5.94% |
| 4 | 10 | 120 | 0.469 | **0.36687** | 0.39012 | 5.96% |
| 4 | 12 | 144 | 0.562 | 0.36746 | 0.39008 | 5.80% |
| 8 | 8 | 192 | 0.750 | 0.36964 | 0.38909 | 5.00% |
| 8 | 10 | 240 | 0.938 | 0.36867 | 0.38899 | 5.22% |
| 8 | 12 | 288 | 1.125 | 0.36866 | 0.38959 | 5.37% |
| 16 | 8 | 384 | 1.500 | 0.37035 | 0.38696 | 4.29% |
| 16 | 10 | 480 | 1.875 | 0.37129 | 0.38594 | 3.80% |
| 16 | 12 | 576 | 2.250 | 0.36926 | 0.38582 | 4.29% |
| 32 | 8 | 768 | 3.000 | 0.37059 | 0.38554 | 3.88% |
| 32 | 10 | 960 | 3.750 | 0.37121 | 0.38497 | 3.57% |
| 32 | 12 | 1,152 | 4.500 | 0.36891 | **0.38467** | 4.10% |

The learned encoder beat FPS at all 12 operating points. Its advantage ranged
from 3.57% to 5.96% Chamfer RMSE.

## The important negative result

The learned model did not form a useful rate-distortion curve. Its best result
was `K=4` at 10 bits per coordinate, and the 96-bit `K=4`, 8-bit result was only
`0.00008` RMSE worse. Increasing the message to 1,152 bits did not improve
distortion. Coordinate precision from 8 to 12 bits also had little effect.

FPS behaved more conventionally: increasing `K` reduced aggregate RMSE from
0.39011 to 0.38467, although three epochs and one seed leave small local
non-monotonicities.

The most likely interpretation is that the current PointNet/max-pool encoder
and folding decoder treat four learned coordinates as a small global latent
code. Extra anchors are pooled away or never receive a distinct role. The
low-dimensional procedural families may make this shortcut especially easy.
This is an inference from the curve, not yet a demonstrated mechanism.

Learned anchor-to-surface RMSE remained between 0.093 and 0.109 across the
sweep and did not improve with `K`. That reinforces the need to visualize
anchors and test whether off-surface displacement is carrying category or
shape parameters.

## Decision

The coordinate-only hypothesis continues to beat the matched FPS control, but
the current architecture fails the stronger requirement that additional rate
buy additional fidelity. It should not be used as the basis of the adaptive
codec unchanged.

Before learning a variable `K` policy, run longer selected-rate controls and
replace global max pooling with a relation-aware set/point transformer. The new
model should first demonstrate a monotonic fixed-rate curve. Then train a
single masked model across `K` values and compare its curve with these
independently trained targets.
