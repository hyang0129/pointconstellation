# Experiments 005-007: three constellation-inference prototypes

Last reviewed: 2026-08-11.

## Purpose

The three implementation options from the
[co-adaptation hypothesis review](co-adaptation-hypotheses.md) now have separate
runnable prototypes. They intervene at different causal points:

1. Experiment 005 changes inference while holding the completion decoder fixed.
2. Experiment 006 constructs a coordinate-code space and decoder together,
   then amortizes held-out coordinate inference.
3. Experiment 007 models a distribution over constellation sets, but remains
   gated on evidence that useful solutions are genuinely multimodal.

These checked-in configurations are CPU smoke tests. They verify contracts,
training paths, metrics, and frozen-model behavior; they are not compression
benchmarks and do not replace the larger diagnostic preflight.

## Shared representation contract

Every path transmits only the final quantized `K x 3` coordinates. Attention
states, gradients, responsibilities, diffusion state, cloud IDs, restart IDs,
and encoder features are local inference machinery and never cross the decoder
boundary.

All three prototypes support a requested `K`; Experiments 005 and 007 also
exercise multiple input sizes in one run. None yet learns the rate-distortion
decision for `K`. A future controller must choose and signal `K`, count those
bits, and retain the raw/pass-through endpoint.

## Experiment 005: competitive semi-amortized refiner

### Implementation

`CompetitiveConstellationRefiner` starts with exchangeable FPS anchors and
applies shared recurrent updates. Each step:

- forms a `K x N` responsibility matrix normalized over anchors so they compete
  for input regions;
- aggregates input evidence into each anchor;
- self-attends across anchors so movements are coordinated;
- optionally consumes a normalized, detached coordinate gradient from the
  frozen decoder; and
- bounds and quantizes the updated coordinates.

The evaluator reports every step in both free-coordinate and strict unique
input-subset modes. The decoder is SHA-256 hashed before and after refiner
training.

```bash
.venv-train/bin/python -m pointconstellation.refiner_experiment \
  --config configs/experiment_005_refiner_smoke.json \
  --device cpu
```

### CPU smoke result

The decoder was unchanged. Free-coordinate validation RMSE improved at every
operating point:

| Input `N` | `K` | Step 0 | Step 2 | Change |
|---:|---:|---:|---:|---:|
| 16 | 4 | 0.38604 | 0.38491 | 0.29% better |
| 16 | 8 | 0.29077 | 0.28969 | 0.37% better |
| 32 | 4 | 0.38987 | 0.38752 | 0.60% better |
| 32 | 8 | 0.28364 | 0.28185 | 0.63% better |

Parameter OOD improved at three of four points. `N=16, K=8` regressed from
`0.31595` to `0.31603`, approximately 0.02%. Strict-subset output was unchanged
in this short run because the small free-coordinate movements did not cross
input-point Voronoi boundaries.

This is the strongest of the three smoke results, but the gains are much too
small to pass the proposed scientific gate. A longer run must include paired
direct optimization, several seeds, analytic-surface metrics, and perturbation
tests.

The subsequent [scale run](experiment-005-refiner-scale-result.md) produced a
material result at `N=256, K=16`: free refinement improved validation RMSE by
22.35%, while strict unique input-subset projection retained a 15.18% gain.
That run strengthens the refiner hypothesis but remains a one-seed result with
important inference-target controls still outstanding.

## Experiment 006: coordinate auto-decoder and amortizer

### Implementation

`LiteralCoordinateBank` contains several independently initialized, learnable
`K x 3` constellations for each training cloud. It supports:

- unrestricted and nearest-input projected coordinate modes;
- straight-through quantization during optimization and exact quantization at
  evaluation;
- alternating coordinate and decoder updates;
- clean, quantization-scale, and multi-scale noisy decoder neighborhoods;
- several held-out coordinate restarts with a bitwise-frozen decoder; and
- a permutation-invariant one-shot amortizer trained by reconstruction plus
  matched-set imitation.

```bash
.venv-train/bin/python -m pointconstellation.auto_decoder_experiment \
  --config configs/experiment_006_auto_decoder_smoke.json \
  --device cpu
```

### CPU smoke result

Held-out refinement kept the decoder unchanged and improved reconstruction
slightly:

| Mode | One-shot RMSE | Refined RMSE | Change | Refined anchor-to-sample RMSE |
|---|---:|---:|---:|---:|
| Unrestricted | 0.54245 | 0.54191 | 0.10% better | 0.22266 |
| Projected | 0.54252 | 0.54195 | 0.11% better | 0.00350 |

The projected mode demonstrates that the surface-proxy constraint is active,
but its held-out coverage RMSE worsened from `0.56793` to `0.71090`. The
unrestricted branch also traded coverage for its small reconstruction gain.
The smoke decoder is intentionally tiny and is not directly comparable to the
Experiment 004 decoder.

This path is implemented but not yet supported as the next large experiment.
It needs a stronger relation-aware decoder, fresh resampling, converged
held-out restarts, and evidence that noise training broadens useful decoder
basins.

## Experiment 007: gated conditional set diffusion

### Implementation

`ConditionalSetDenoiser` is invariant to input order and equivariant to
constellation-particle order. It conditions on the input cloud, timestep, and
requested `K`. The runner:

1. trains and freezes a shared completion decoder;
2. optimizes several quantized constellation restarts per training cloud;
3. measures matched-set separation among restarts with comparable distortion;
4. trains a denoiser on the best optimized targets;
5. starts reverse sampling from noised FPS rather than pure Gaussian noise; and
6. selects the best of several final quantized candidates through the frozen
   decoder.

The runner executes even if its multimodality gate fails, but labels that result
`experimental_not_justified`.

```bash
.venv-train/bin/python -m pointconstellation.diffusion_experiment \
  --config configs/experiment_007_set_diffusion_smoke.json \
  --device cpu
```

### CPU smoke result

The decoder was frozen and bitwise unchanged. Denoising noise MSE decreased
from `1.07199` to `0.94935`. The configured gate marked 79.17% of shallow
restart groups as multimodal, but those targets received only six optimization
steps. Unconverged restarts can look like distinct modes, so the runner records
that a convergence audit is still required.

Diffusion was worse than FPS at every validation operating point:

| Input `N` | `K` | Diffusion best-of-3 RMSE | FPS RMSE | Gap |
|---:|---:|---:|---:|---:|
| 24 | 4 | 0.48343 | 0.46744 | 3.42% worse |
| 24 | 8 | 0.42359 | 0.36159 | 17.15% worse |
| 32 | 4 | 0.49158 | 0.46743 | 5.17% worse |
| 32 | 8 | 0.43002 | 0.36568 | 17.59% worse |

The implementation is therefore runnable but currently uncompetitive. A
larger run is not justified until converged restarts preserve the multimodality
signal.

## Verification

The combined repository has 50 passing tests: the original 28 plus six
refiner, eight auto-decoder, and eight diffusion tests. The new tests cover:

- variable `N` and `K` shapes;
- input invariance and constellation equivariance;
- competitive responsibility normalization;
- exact final quantization;
- strict unique input-subset projection;
- finite gradients;
- multiple coordinate restarts;
- best-of-candidate decoder scoring;
- multimodality gate pass and fail behavior;
- frozen and bitwise-unchanged decoders; and
- tiny end-to-end CLI runs that write checkpoints and metrics.

## Current decision

Experiment 005 remains the recommended next scaling target. It changes
inference without changing decoder capacity and already moves validation in the
expected direction. Experiment 006 is a fallback if the existing decoder's
basins prove brittle. Experiment 007 should remain a gated research branch.
