# Experiment 004: frozen-decoder bottleneck audit result

## Outcome

The decoder passed its rate-utilization checks, the progressive learned subset
failed its matched-FPS checks, and the free-coordinate oracle exposed material
off-sample coordinate-coding headroom.

This separates the failures that Experiment 003 could not distinguish:

- completion from coordinate subsets works and improves strongly with `K`;
- the current learned selection mechanisms do not discover useful distributed
  constellations; and
- coordinates optimized through the frozen decoder contain more useful
  information than the sampled subsets tested here, but cease to be a strict
  input-subset representation.

The overall gate failed because the learned selector remained far behind FPS.

## Run

- Apple MPS
- PyTorch 2.13.0
- 256-point dense targets
- input sizes `{64, 128, 256}`
- `K` in `{4, 8, 16, 32}`
- 12-bit coordinates
- 448 training, 140 validation, and 140 parameter-OOD clouds
- 12 completion-decoder epochs and 12 selector epochs
- selector checkpoint chosen on validation only: epoch 11
- 240,387 decoder and 187,105 selector parameters
- 42.33 seconds for the final local run

The frozen decoder was bitwise unchanged during selector training.

## Primary operating point

The primary point is input `N=256`, `K=16`, and a 576-bit raw coordinate
payload.

| Condition | Validation RMSE | OOD RMSE | Validation coverage RMSE | Anchor-to-target-sample RMSE | Validation Hausdorff |
|---|---:|---:|---:|---:|---:|
| FPS | **0.18261** | **0.16873** | 0.24438 | 0.00025 | 0.54500 |
| Random | 0.19193 | 0.17513 | 0.30831 | 0.00025 | 0.57881 |
| Learned progressive subset | 0.25866 | 0.25840 | 0.69887 | 0.00025 | 0.65360 |
| Best of FPS + 8 random subsets | 0.16820 | 0.15687 | 0.28410 | 0.00025 | 0.55934 |
| Free-coordinate oracle | **0.14196** | **0.13372** | 0.27397 | 0.08374 | **0.51824** |

All subset conditions remained on the quantized input surface. Received-point
preservation RMSE was approximately `0.0001` for every condition, at the
numerical floor of the pairwise-distance implementation, because transmitted
coordinates are copied into the reconstruction.

The best sampled subset beat FPS by 7.89% on validation and 7.03% on OOD. A
better strict subset therefore exists even in this small random candidate pool;
the learned selector simply did not find it.

The separately evaluated free-coordinate condition is 15.60% below the best
sampled-subset condition on validation and 14.76% below it on OOD. The two
conditions use different random candidate pools, so those percentages are not
paired before/after optimization gains. Its surface RMSE increased from
approximately `0.00025` to `0.08374`, which is evidence for coordinates away
from the finite observed samples rather than a better strict subset. The metric
is nearest-neighbor distance to the 256-point target, not analytic distance to
the underlying procedural surface; it does not establish that the coordinates
are truly off-surface.

The best sampled subset is the best of FPS and only eight random candidates.
Its error is a feasible reference, not the optimum achievable by an input
subset.

## Frozen-decoder FPS rate curve

| `K` | Payload bits | Validation RMSE | OOD RMSE |
|---:|---:|---:|---:|
| 4 | 144 | 0.24198 | 0.22075 |
| 8 | 288 | 0.20877 | 0.19830 |
| 16 | 576 | 0.18261 | 0.16873 |
| 32 | 1,152 | 0.14890 | 0.13261 |

Validation improved by 38.47% and OOD by 39.93% from `K=4` to `K=32`. Every
adjacent rate improved. This is the first experiment in which one shared
decoder, rather than independently trained per-rate models, exhibits a clean
monotonic coordinate-rate curve.

## Learned progressive subset rate curve

| `K` | Validation learned | Validation FPS | OOD learned | OOD FPS |
|---:|---:|---:|---:|---:|
| 4 | 0.38489 | 0.24198 | 0.36289 | 0.22075 |
| 8 | 0.30844 | 0.20877 | 0.29904 | 0.19830 |
| 16 | 0.25866 | 0.18261 | 0.25840 | 0.16873 |
| 32 | 0.23813 | 0.14890 | 0.23547 | 0.13261 |

The learned ranking itself improves with `K`, but it is 41.65% worse than FPS
on validation and 53.14% worse on parameter-OOD data at the primary point.

Its validation coverage RMSE is `0.69887`, compared with `0.24438` for FPS.
Hard uniqueness removed literal anchor duplication, but the scalar ranking
still selects points from too little of the cloud. Reconstruction gradients
through the straight-through ranking were not sufficient to teach global
allocation.

## Variable input size

At fixed `K=16`:

| Input `N` | Validation FPS | Validation learned | OOD FPS | OOD learned |
|---:|---:|---:|---:|---:|
| 64 | 0.17697 | 0.22041 | 0.16564 | 0.21739 |
| 128 | 0.18145 | 0.23482 | 0.16738 | 0.23575 |
| 256 | 0.18261 | 0.25866 | 0.16873 | 0.25840 |

Both models accept all configured input sizes. The selector gets worse as its
candidate pool grows, another sign that independent importance ranking is the
wrong allocation mechanism. FPS changes only slightly.

## Gate

| Check | Threshold | Result | Status |
|---|---:|---:|---|
| Validation FPS endpoint improvement | at least 1% | 38.47% | Pass |
| OOD FPS endpoint improvement | at least 1% | 39.93% | Pass |
| Largest adjacent FPS regression | at most 0.5% | none | Pass |
| Validation learned gap versus FPS | at most 5% | 41.65% worse | Fail |
| OOD learned gap versus FPS | at most 5% | 53.14% worse | Fail |
| Decoder unchanged | required | bitwise unchanged | Pass |

The diagnostic 5% free-coordinate-headroom threshold also triggered on both
splits, but it is not part of the pass/fail gate.

## Interpretation and next test

The broader diagnosis is recorded in
[Co-adaptation and constellation inference
hypotheses](co-adaptation-hypotheses.md). It distinguishes the directly
observed subset-selection gap from the larger hypothesis class available to
the off-sample free-coordinate oracle and surveys relevant work in amortized
inference, auto-decoders, set prediction, discrete bottlenecks, emergent
communication, and coupled optimization.

Do not replace the completion decoder with diffusion yet. The deterministic
decoder already uses coordinate rate correctly, preserves received points, and
generalizes to held-out parameter ranges.

Do not proceed directly to adaptive `K` either. A rate controller cannot repair
a selector that chooses a poor subset at every fixed rate.

The next experiment should first validate and decompose the apparent
free-coordinate advantage. Pair each optimization with its exact initialization,
run multiple starts and inference depths, measure quantization and true-surface
sensitivity, and compare scalar subset logits, competitive assignment,
surface-constrained coordinates, and unrestricted coordinates against the same
frozen decoder.

If that audit confirms robust per-cloud headroom, the recommended implementation
is a competitive semi-amortized constellation refiner: a permutation-aware
encoder initializes `K` exchangeable anchors, and a shared recurrent update
uses input responsibilities plus decoder residual or gradient information to
improve them. The full controls, alternatives, and stop/go criteria are in the
[hypothesis note](co-adaptation-hypotheses.md#three-implementation-options).
