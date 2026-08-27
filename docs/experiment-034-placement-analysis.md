# Experiment 034: semantic placement analysis

Status: complete for decoder seed 7 / refiner seed 101; Gate H-B4 passes narrowly (Adam-64 free coordinates are more category-consistent than strict subsets; the refiner is not).

## Question and hypothesis

Experiment 034 tests H-B4: where do the transmitted constellation coordinates
land in the canonical object frame? The diagnostic hypothesis is that
free-coordinate inference produces category-consistent placement beyond what is
already explained by source-geometry subset controls.

The predeclared primary statistic is the ratio of mean within-category to mean
across-category constellation distance. The distance between two unordered
constellations is their symmetric mean nearest-neighbour Euclidean distance.
A lower ratio means that different instances of one category place their
coordinates more similarly than instances from different categories. The H-B4
gate passes only if both the refiner and Adam-64 ratios are lower than both FPS
and random-best-of-16 ratios. This is deliberately stronger than showing that
canonical ModelNet geometry is category-consistent.

The gate is descriptive for one sealed model cell. Passing it would motivate a
multi-seed confirmatory run; it would not show semantic decoding, competitive
compression, or generalization beyond ModelNet40. Failing it would retain the
simpler explanation that placement follows source geometry or idiosyncratic
per-cloud optimization.

## Fixed protocol

- Reuse the Experiment 019 stabilized decoder for seed 7 and its assigned
  refiner checkpoint for seed 101. Both are selected using calibration data and
  remain frozen. The summary labels the single-cell scope explicitly.
- Evaluate all four official-test instances in each of the 32 training
  categories and all four instances in each of the eight held-out categories.
  The latter retain the `ood` label. No validation or OOD value selects a model.
- Preserve the ModelNet40 manifest frame: mesh bounding-box centering followed
  by unit maximum-vertex-radius scaling. Do not rotate, reflect, or register
  individual instances after normalization.
- Compare four `K=8`, 12-bit coordinate messages: deterministic FPS,
  random-best-of-16, the recurrent refiner, and source-only Adam/STE with 64
  decoder evaluations initialized from FPS. Random-best-of-16 reuses the
  Experiment 021 seed family and selects a strict subset by frozen-decoder
  source Chamfer. Adam-64 and the refiner are free-coordinate methods.
- Every output is serialized with the canonical fixed-width bitstream and
  decoded before analysis. Each message is exactly 50 bytes including its
  14-byte header. The JSONL stores the stream, SHA-256, decoded coordinates,
  actual bits, and actual bits per source point.
- The encoder-visible source sample is the only target used by selection,
  recurrent inference, or Adam. No normals, fresh sample, labels, part IDs, or
  target-only information enter the message.

## Placement maps and entropy

For each category and method, coordinates from its four instances are
accumulated into a fixed `8 x 8 x 8` histogram over `[-1, 1]^3`. The summary
stores the sparse voxel counts, occupied-voxel count, and Shannon entropy in
bits. It also reports entropy normalized by the full grid-volume maximum and by
the maximum support possible for the observed point count. The latter is useful
for small `4 x K` category samples, but neither normalization corrects finite
sample bias.

`category_placement_maps.svg` shows the x-z marginal of these maps for every
available category and all four methods. The JSON histogram, not the projected
SVG, is the quantitative three-dimensional map.

## Part labels and geometry proxies

No ShapeNet-Part or other compatible part-labelled subset was present in the
local data checkout when this experiment was implemented. ModelNet40 instance
IDs also do not provide a direct join to ShapeNet part annotations. The primary
runner therefore records `part_hit_analysis.fallback =
mesh_geometry_proxies` and evaluates these per-coordinate diagnostics:

- curvature percentile: the midrank percentile of area-weighted incident-face
  normal dispersion at the nearest normalized-mesh vertex;
- symmetry-plane distance: absolute coordinate distance to the best of the
  three canonical coordinate planes, selected by the mean finite-vertex
  distance after reflection and normalized by mesh extent on that axis; and
- extremity score: radial coordinate norm divided by the maximum normalized
  mesh-vertex radius.

These quantities are proxies. The selected coordinate plane need not be the
object's true semantic symmetry plane, and nearest-vertex curvature is not a
continuous differential curvature estimate. The module also exposes a tested
radius-based part-hit function for a future compatible labelled subset; it is
not used for the ModelNet40 result.

The runner additionally stores each coordinate's nearest distance to the
finite, encoder-visible source sample. It is named
`nearest_source_sample_distance` throughout and must not be described as
distance to the underlying continuous surface.

## Qualitative figure

`qualitative_overlays.svg` is a headless vector figure with source, FPS,
random-best-of-16, refiner, and Adam-64 columns for eight fixed categories:
airplane, car, chair, guitar, lamp, sofa, bed, and table. Bed and table are
held-out-category OOD examples. Each constellation is overlaid on its source
sample in the canonical x-z projection. Constellation colors encode nearest
source-sample distance with a shared robust scale. The figure is qualitative;
occlusion in the two-dimensional projection is not a placement metric.

## Outputs and checks

The runner writes:

- `run_manifest.json`: exact config, Experiment 019 input hashes, data protocol,
  model artifact hashes, and canonical-frame declaration;
- `placement_points.jsonl`: one exact serialized message and its placement
  diagnostics for every cloud and method;
- `placement_summary.json`: category maps, entropy, consistency ratios, proxy
  summaries, the H-B4 gate, scope limitations, and contract checks;
- `category_placement_maps.svg`: all-category projected density maps; and
- `qualitative_overlays.svg`: the eight-category comparison figure.

Resume is permitted only when the manifest is byte-for-byte equivalent after
JSON parsing. Duplicate cloud/method rows and incomplete four-method grids are
rejected. The runner verifies Experiment 019 data identity and contract checks,
canonical stream round trips, and the frozen decoder state hash.

## Reproduction

The full analysis requires the ignored Experiment 019 checkpoints, local
ModelNet40 manifest and meshes. Run it from a checkout containing those inputs:

```bash
.venv-train/bin/python -m pointconstellation.placement_analysis \
  --config configs/experiment_034_placement_analysis.json \
  --device mps
```

The command performs no training, but the 64-evaluation per-cloud search is not
a smoke test and was not run as part of this implementation change. The
synthetic consistency metric and both SVG generators are covered by the normal
unit suite:

```bash
.venv-train/bin/python -m pytest -q tests/test_placement_analysis.py
```

## Results

Executed on homen-linux (RTX 5070 Ti, CUDA) in under a minute for decoder seed 7 / refiner seed 101 over all
160 validation and category-OOD clouds (40 categories). No part-labelled subset was
available, so placement is characterised by category consistency and mesh-geometry proxies only.

**Gate H-B4 passes, but narrowly.** Criterion: both free-coordinate methods must have a lower within/across
set-distance ratio than both strict-subset controls.

| Method | Class | Within-category mean | Across-category mean | Within/across ratio | Relative separation |
|---|---|---:|---:|---:|---:|
| `fps` | strict-subset | 0.3733 | 0.4771 | 0.7825 | 0.2175 |
| `random_best_of_16` | strict-subset | 0.3745 | 0.4766 | 0.7857 | 0.2143 |
| `refiner` | free-coordinate | 0.3746 | 0.4798 | 0.7807 | 0.2193 |
| `adam_64` | free-coordinate | 0.4355 | 0.5838 | 0.7460 | 0.2540 |

### Reading

1. Decoder-aware **Adam-64 constellations are modestly more category-consistent** than FPS or decoder-guided
   random subsets (ratio 0.746 vs 0.782): free coordinates chosen against the decoder land in more
   repeatable places for instances of the same category than any strict subset does.
2. The **refiner's constellations are not more consistent than FPS** (0.7807 vs 0.7825, a difference far inside
   noise), so the gate passes on the strength of the Adam arm alone. This mirrors Experiments 021/022: the
   refiner behaves like FPS-plus-small-corrections, whereas full decoder-aware search moves points substantially.
3. Absolute separation is weak for every method (within/across ratios 0.75--0.79): eight points in a canonical
   frame carry category information, but far less than the classifier results of Experiment 030 would suggest
   is available in the raw coordinates — the set-distance statistic is a blunt instrument here.
4. Part-hit rates remain untested (`part_hit_analysis.status = not_used_in_modelnet40_analysis`); the tested
   radius-based metric is ready for a ShapeNet-Part-compatible subset.

Artifacts: `artifacts/local/experiment_034_placement_analysis/placement_summary.json`, `placement_points.jsonl`,
`category_placement_maps.svg`, `qualitative_overlays.svg`.

