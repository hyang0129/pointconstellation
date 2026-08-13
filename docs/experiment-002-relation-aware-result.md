# Experiment 002: relation-aware selected-rate result

Run date: 2026-08-10

Experiment 002 tested whether preserving relationships between constellation
points makes additional coordinate rate useful. It also repeated the legacy
model with substantially more training and evaluated a matched FPS encoder with
the relation-aware decoder.

The experiment followed the predefined
[plan and gate](experiment-002-relation-aware-plan.md) without changing its
thresholds after observing results.

## Configuration

- Apple M5 MPS, PyTorch 2.13.0
- 256 input/output points
- `K in {4, 16, 32}` at 12 bits per coordinate
- 448 training, 140 validation, and 140 parameter-OOD clouds
- 12 epochs, seed 7, batch size 8
- identical loss weights and optimizer settings across rates
- nine independently trained model/rate points
- 203.41 seconds total wall time

Command:

```bash
.venv-train/bin/python -m pointconstellation.selected_rate \
  --config configs/experiment_002_relation_aware.json \
  --device mps
```

## Aggregate result

| Model | `K` | Payload bits | Parameters | Validation RMSE | OOD RMSE | Validation anchor-surface RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Legacy learned | 4 | 144 | 251,663 | **0.15565** | **0.15491** | 0.08656 |
| Relation learned | 4 | 144 | 437,574 | 0.18303 | 0.18652 | 0.07578 |
| Relation FPS | 4 | 144 | 230,883 | 0.17434 | 0.16746 | 0.00025 |
| Legacy learned | 16 | 576 | 260,915 | 0.15086 | 0.15005 | 0.08620 |
| Relation learned | 16 | 576 | 438,726 | 0.17456 | 0.17969 | 0.07089 |
| Relation FPS | 16 | 576 | 230,883 | **0.12665** | **0.11687** | 0.00025 |
| Legacy learned | 32 | 1,152 | 273,251 | 0.14833 | 0.14959 | 0.08738 |
| Relation learned | 32 | 1,152 | 440,262 | 0.16959 | 0.17468 | 0.07122 |
| Relation FPS | 32 | 1,152 | 230,883 | **0.11850** | **0.11087** | 0.00025 |

Per-family validation and OOD metrics are retained in the generated
`selected_rate.json` artifact.

## Gate result: pass

The relation-aware learned curve passed both predefined requirements:

| Split | K=4 RMSE | K=16 RMSE | K=32 RMSE | Endpoint improvement | Largest adjacent regression | Result |
|---|---:|---:|---:|---:|---:|---|
| Validation | 0.18303 | 0.17456 | 0.16959 | 7.35% | 0.00% | Pass |
| Parameter OOD | 0.18652 | 0.17969 | 0.17468 | 6.35% | 0.00% | Pass |

The required endpoint improvement was 1%, and an adjacent regression above
0.5% would have failed the gate.

## What changed from Experiment 001

The longer-trained legacy model also formed a monotonic curve: it improved
4.70% on validation and 3.43% on parameter OOD from `K=4` to `K=32`. Its
K=4 validation RMSE was 0.15565, compared with approximately 0.367 in the
three-epoch sweep.

Therefore, Experiment 001's flat learned curve was substantially confounded by
undertraining. Global pooling may still limit how anchors are used, but the
earlier sweep alone did not demonstrate that mechanism. Selected-rate curves
must be trained near convergence before drawing architectural conclusions.

## The important control result

The relation-aware decoder clearly uses additional geometric points: its FPS
curve improved 32.03% on validation and 33.79% on OOD from `K=4` to `K=32`.
At K=16 and K=32 it was the best tested model by a large margin.

The learned relation encoder was not competitive. It was 14.3-17.6% worse than
the legacy learned model across the three validation rates, and it was 5.0%,
37.8%, and 43.1% worse than relation FPS at K=4, K=16, and K=32 respectively.
Its anchors remained 0.058-0.076 units from the input surface on aggregate,
while quantized FPS anchors were effectively on-surface.

This isolates the current weakness primarily to anchor generation rather than
the relation-aware decoder. The learned encoder is still using off-surface
coordinate displacement and is failing to select a constellation as useful as
ordinary farthest-point samples.

The parameter-OOD split is a distribution shift, not a guaranteed harder set;
normalization can make its aggregate RMSE lower than validation. Its value here
is the consistency of the rate trend, not the absolute ordering of the splits.

## Decision

Accept the relation-aware decoder as a viable fixed-rate component, but do not
move directly to adaptive `K`. The primary rate-utilization gate passed, while
the matched encoder control exposed a more immediate problem.

The next encoder isolation should keep the relation decoder fixed and compare:

1. FPS anchors;
2. the current soft-projected learned proposals;
3. learned hard input-subset selection; and
4. projection temperature and surface-loss sweeps.

Only after a learned encoder approaches or beats relation FPS at matched rate
should the selected-rate result be repeated across multiple seeds and promoted
to the masked multi-`K` adaptive experiment.
