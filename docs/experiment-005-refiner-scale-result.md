# Experiment 005: competitive refiner scale result

## Outcome

Scaling the competitive semi-amortized refiner turned the small smoke signal
into a material result. After correcting encoder-side decoder-gradient feedback
to use only the available input cloud, the refiner improved every tested
validation and parameter-OOD operating point in free-coordinate mode.
Projecting its final coordinates to a strict unique input subset retained most
of the gain.

At the primary `N=256`, `K=16`, 12-bit operating point:

| Condition | Validation RMSE | OOD RMSE | Validation anchor-to-sample RMSE |
|---|---:|---:|---:|
| FPS / refinement step 0 | 0.18261 | 0.16873 | 0.00025 |
| Refined free coordinates, step 8 | **0.14073** | **0.13252** | 0.07777 |
| Refined strict unique input subset, step 8 | **0.15625** | **0.14405** | 0.00025 |

Free refinement improved validation by 22.94% and parameter OOD by 21.46%.
Strict-subset projection improved validation by 14.44% and parameter OOD by
14.63%.

The free result is approximately level with the short per-cloud coordinate
optimizer from Experiment 004 (`0.14196` validation), while the strict result
beats that experiment's best-of-nine sampled subset (`0.16820`). This is
evidence that decoder-conditioned competitive updates can find useful
constellations that scalar ranking and a small random search do not.

## Run

- Apple MPS
- existing Experiment 004 decoder checkpoint
- decoder SHA-256 unchanged before and after refiner training
- 448 training, 140 validation, and 140 parameter-OOD clouds
- 256-point reconstruction targets
- input candidate sizes `{64, 128, 256}`
- `K` in `{4, 8, 16, 32}`
- 12-bit coordinates
- eight shared refinement steps
- 24 refiner epochs
- 205.86 seconds for the corrected gradient-feedback arm
- 136.15 seconds for the matched no-feedback arm

The checked-in configuration is
[`experiment_005_refiner_scale.json`](../configs/experiment_005_refiner_scale.json).
The matched ablation is
[`experiment_005_refiner_scale_no_decoder_gradient.json`](../configs/experiment_005_refiner_scale_no_decoder_gradient.json).
The saved decoder can be supplied without retraining it:

```bash
.venv-train/bin/python -m pointconstellation.refiner_experiment \
  --config configs/experiment_005_refiner_scale.json \
  --decoder-checkpoint artifacts/local/experiment_004_frozen_decoder/decoder.pt \
  --device mps
```

## Primary refinement trajectory

| Step | Validation free | Validation strict subset | OOD free | OOD strict subset |
|---:|---:|---:|---:|---:|
| 0 | 0.18261 | 0.18261 | 0.16873 | 0.16873 |
| 1 | 0.16007 | 0.17143 | 0.14918 | 0.15658 |
| 2 | 0.15278 | 0.16404 | 0.14346 | 0.15160 |
| 3 | 0.14906 | 0.15981 | 0.13927 | 0.14734 |
| 4 | 0.14620 | 0.15965 | 0.13682 | 0.14659 |
| 5 | 0.14370 | 0.15704 | 0.13495 | 0.14526 |
| 6 | 0.14247 | 0.15831 | 0.13406 | 0.14506 |
| 7 | 0.14134 | 0.15685 | 0.13288 | 0.14437 |
| 8 | **0.14073** | **0.15625** | **0.13252** | **0.14405** |

Free-coordinate validation and OOD improvement is monotonic across all eight
steps. Strict projection is mostly monotonic, with small reversals caused by
crossing input-point Voronoi boundaries.

## Decoder-gradient ablation

The same run without decoder-gradient feedback still improves the primary
point, but by much less:

| Arm | Validation free | Validation strict | OOD free | OOD strict |
|---|---:|---:|---:|---:|
| FPS / step 0 | 0.18261 | 0.18261 | 0.16873 | 0.16873 |
| No decoder gradient | 0.17334 | 0.17726 | 0.16065 | 0.16259 |
| Input-only decoder gradient | **0.14073** | **0.15625** | **0.13252** | **0.14405** |

The recurrent competitive model alone yields 5.07% free-coordinate validation
improvement. Legal decoder feedback raises that to 22.94%, so the dominant
factor at this operating point is useful frozen-decoder gradient information,
with a smaller independent contribution from learned allocation/refinement.

## Rate and input-size behavior

Every validation combination of candidate size and `K` improved after eight
steps. At `N=256`:

| `K` | Step-0 FPS | Refined free | Refined strict subset |
|---:|---:|---:|---:|
| 4 | 0.24198 | 0.19282 | 0.20676 |
| 8 | 0.20877 | 0.16526 | 0.17883 |
| 16 | 0.18261 | 0.14073 | 0.15625 |
| 32 | 0.14890 | 0.11047 | 0.12656 |

The refined curves remain monotonic in `K`. Larger candidate pools generally
help strict projection, unlike the scalar selector in Experiment 004.

## Interpretation

This is strong support for a combined inference and allocation explanation:

- a one-shot scalar ranking was not finding the decoder's useful messages;
- competitive responsibilities let anchors acquire different spatial roles;
- recurrent updates provide a path between constellation arrangements; and
- decoder-gradient feedback supplies direct information about what the current
  constellation fails to reconstruct.

The strict-subset result matters because it cannot hide arbitrary features in
off-sample coordinates. The free result still moves away from the observed
sample and should be evaluated against the analytic procedural surface before
being treated as geometric rather than coded.

## Limitations and next controls

This is not yet a passed compression result:

1. It is one refiner seed against one decoder checkpoint.
2. The runner starts from FPS, so it does not yet compare learned and direct
   initializers.
3. Decoder-gradient feedback now uses only the candidate input set. An earlier
   run incorrectly supplied the complete target for partial-input rows; its
   numbers were discarded and the checked-in artifact was regenerated.
4. Strict subset is a post-hoc unique nearest-input projection, so its gain is
   projection headroom rather than proof of a learned discrete selector.
5. The result lacks analytic-surface distance, fresh resampling, quantization
   perturbation, direct-optimizer, and multi-seed controls.
6. Runtime and transmitted cardinality metadata are not yet part of a coded
   rate comparison.

The next scale run should preserve the primary configuration while adding
three seeds, gradient-free and direct-gradient inference bounds, target-equals-
input variable-`N` evaluation, and the five additional fan-out hypotheses.
