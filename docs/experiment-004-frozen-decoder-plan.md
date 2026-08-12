# Experiment 004: frozen-decoder bottleneck audit

Status: implemented and run locally. See the
[Experiment 004 result](experiment-004-frozen-decoder-result.md).

## Question

Experiment 003 allowed the selector and decoder to adapt to one another. Its
learned hard subset stayed on the input surface but was substantially worse
than FPS, while every soft-query encoder collapsed. That result does not tell
us whether the failure is in the coordinate representation, the selector, or
joint optimization.

Experiment 004 asks:

> With one completion decoder held exactly fixed, how much reconstruction
> information can FPS, a learned progressive subset, an optimized input
> subset, and unrestricted coordinates communicate at the same `K x 3` rate?

The experiment separates completion learning from constellation selection. It
is a bottleneck audit, not yet an adaptive rate controller.

## Stage A: variable-`K` completion decoder

Train one relation-aware decoder using only quantized coordinate subsets made
outside the learned encoder. Each training batch chooses:

- an input cardinality from the configured input-size set;
- `K` from `{4, 8, 16, 32}`; and
- FPS, uniform random, or normal-aware FPS sampling.

The decoder receives no point features, sample indices, normals, sampler
identity, or input cardinality. Constellation cardinality is available because
it is part of the counted bitstream syntax. The reconstruction copies every
transmitted coordinate verbatim and generates only the remaining output
points. Consequently, increasing reconstruction density cannot move or erase
the received geometry.

The same decoder handles every configured `K`; no per-rate decoder is trained.

## Stage B: frozen progressive subset encoder

Freeze all decoder parameters and train one permutation-equivariant transformer
to assign a scalar importance score to each input point. Sorting those scores
defines a nested sequence of subsets. Training uses a straight-through hard
selection: the forward message is always a unique input subset, while gradients
follow temperature-controlled soft assignments.

Training alternates input sizes and `K` values. No query identity, ranking,
score, or attention feature crosses the bottleneck. The selected coordinates
are quantized identically to all controls.

An exact decoder state comparison before and after selector training is a
required integrity check.

## Frozen-decoder controls

Evaluate the following through the identical frozen decoder:

1. deterministic FPS;
2. deterministic seeded random subsets;
3. the learned progressive subset;
4. best-of-sampled input subsets, using FPS plus multiple random trials as a
   practical on-surface upper-bound probe; and
5. free coordinates initialized from the best input subset and optimized per
   cloud through the frozen decoder.

The best-of-sampled condition is not claimed to be the exact combinatorial
on-surface optimum. The free-coordinate condition is also an analysis oracle,
not a deployable encoder. Together they measure whether the representation has
unused subset-selection headroom or instead benefits mainly from arbitrary
coordinate coding.

## Configuration

- 256-point dense reconstruction target
- input sizes `{64, 128, 256}`
- constellation sizes `{4, 8, 16, 32}`
- 12 bits per coordinate
- 448 training, 140 validation, and 140 parameter-OOD clouds
- 12 decoder epochs and 12 frozen-decoder selector epochs
- batch size 8 and seed 7
- primary gate at input `N=256`, `K=16`

This remains a one-seed architectural screen. A passing result requires a
later multi-seed confirmation and actual entropy-coded rate measurement.

## Measurements

For every normal control and rate, record:

- symmetric Chamfer RMSE and Hausdorff distance;
- input-to-constellation coverage RMSE;
- constellation-to-surface RMSE;
- received-point preservation RMSE;
- mean minimum anchor separation;
- coordinate payload bits;
- validation, parameter-OOD, and per-family metrics; and
- encoder/decoder parameter counts and wall time.

The two oracle probes are run only at the primary input size and rate to bound
local runtime.

The final evaluation uses two intersecting slices rather than the complete
Cartesian product: all `K` values at the primary input size, and all input sizes
at the primary `K` value.

## Predeclared gate

The audit passes only if:

1. frozen-decoder FPS improves from `K=4` to `K=32` by at least 1% on both
   validation and parameter-OOD data;
2. no adjacent FPS rate regresses by more than 0.5%;
3. the learned progressive subset is no more than 5% worse than FPS at the
   primary validation and parameter-OOD operating point; and
4. the decoder state is bitwise unchanged during selector training.

The free-coordinate improvement over the best sampled input subset is reported
as a diagnostic, not a pass criterion. An improvement of at least 5% is treated
as evidence that arbitrary coordinate coding has material headroom over the
strict geometric-subset hypothesis.

Passing authorizes explicit per-cloud rate selection over the learned nested
subsets plus a raw/pass-through candidate. Failing the FPS rate checks means the
completion decoder is inadequate. Passing the decoder checks but failing the
learned-subset check keeps the work focused on selection rather than changing
the decoder or adding diffusion.

## Command

```bash
.venv-train/bin/python -m pointconstellation.bottleneck_audit \
  --config configs/experiment_004_frozen_decoder.json \
  --device mps
```

Use `--resume` to reuse completed decoder or selector checkpoints.
