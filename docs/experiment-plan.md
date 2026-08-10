# Experiment plan

## Research question

At an equal *actual coded rate*, can an unordered, quantized `K x 3`
constellation decoded by a shared model reconstruct useful point-cloud geometry
as well as or better than established codecs and simpler sampling baselines?

The initial target is lossy static geometry without color. Attributes, temporal
coding, and lossless reconstruction are out of scope until this question is
answered.

The intended codec is not permanently fixed-rate. It must eventually accept a
variable number of input points, select a variable constellation size, and
produce a variable-size reconstruction. A raw/pass-through mode is a required
endpoint when constellation coding does not improve the rate-distortion trade.
The [adaptive codec target](adaptive-codec.md) defines this direction.

## Milestone 0: analytic primitives

Validate the representation contract with deterministic codecs:

- plane: four rectangle corners;
- line segment: two endpoints;
- sphere: a center plus axis points, while documenting the ambiguity of an
  untyped constellation; and
- mixtures: test whether grouping can be inferred without transmitting patch
  IDs.

Success means stable permutation-invariant decoding and transparent distortion
measurements. It does not establish learned compression performance.

## Milestone 1: learned fixed-rate autoencoder (current)

Use a fixed input size `N` and constellation size `K`.

### Encoder

1. Build local geometric features with an E(3)-equivariant point network.
2. Select or regress `K` anchor coordinates.
3. Project or softly constrain anchors near the input surface.
4. Quantize coordinates with straight-through gradients and add lattice-scale
   jitter.
5. Discard every feature channel at the bottleneck.

### Decoder

1. Accept only quantized anchor coordinates.
2. Construct a k-nearest-neighbor graph from their relative positions.
3. Use an E(3)-equivariant upsampling network to predict dense points.
4. Produce a fixed `N` initially; later treat output density as a renderer
   setting.

### Loss

```text
L = distortion(X, X_hat)
  + lambda_surface * anchor_surface_distance(Z, X)
  + lambda_repulse * anchor_repulsion(Z)
  + lambda_equiv * equivariance_error
  + lambda_rate * estimated_bits(Z_quantized)
```

Begin with symmetric Chamfer distance for iteration speed. Final evaluation
must include point-to-point and point-to-plane distortion, Hausdorff or tail
error, normal consistency, and task metrics where relevant.

## Baselines and ablations

Use identical train/test data and preprocessing.

| Question | Comparison |
|---|---|
| Does learning help? | random sampling, voxel-grid sampling, farthest-point sampling |
| Does a decoder help? | sampled points alone versus sampled points plus the same decoder |
| Do pure coordinates suffice? | `K x 3` versus `K x (3 + F)` at matched coded bits |
| Is geometry meaningful? | unrestricted anchors versus input-subset and surface-proximal anchors |
| Does precision hide features? | float32, 16-, 12-, 10-, and 8-bit coordinate lattices plus jitter |
| Does equivariance help? | equivariant model versus augmentation-only model |
| Is it competitive? | Draco, MPEG G-PCC, and JPEG Pleno where runnable reference code is available |

Compare rate-distortion curves, not a single `K`.

The first fixed-rate sweep is recorded in the
[rate sweep report](experiment-001-rate-sweep.md). The learned model beat FPS
at every tested point but produced an essentially flat curve, so a
relation-aware decoder is required before adaptive-`K` training.

The next architectural gate is defined in the
[Experiment 002 relation-aware plan](experiment-002-relation-aware-plan.md).
It compares longer legacy, relation-aware learned, and matched-decoder FPS
curves at selected rates on both in-distribution and parameter-OOD geometry.
The [local result](experiment-002-relation-aware-result.md) passed the
rate-utilization gate, corrected the earlier undertraining confound, and
isolated learned anchor generation as the next bottleneck.

[Experiment 003](experiment-003-encoder-isolation-plan.md) holds the
relation-aware decoder controls constant and compares FPS, soft projection,
and learned hard input-subset selection before adaptive-cardinality work.
Its [local result](experiment-003-encoder-isolation-result.md) failed the
predeclared gate and identified anchor collapse and missing input coverage as
the next encoder objective.

## Dataset ladder

1. Synthetic planes, corners, cylinders, spheres, thin structures, and mixtures.
2. Held-out procedural combinations to expose primitive memorization.
3. CAD/object surfaces such as ShapeNet or ModelNet, subject to their licenses.
4. Indoor scenes for the wall/floor prior.
5. Outdoor LiDAR only after the model handles variable density and scale.

Splits must hold out shape instances and procedural parameter ranges. A second
split should hold out entire primitive combinations or semantic categories.

## Metrics

### Rate

- actual total bits and bits per input point;
- coordinate precision and entropy-coding configuration;
- all per-cloud metadata;
- decoder binary/model size, reported separately and amortized at several
  collection sizes.

### Distortion

- D1-like point-to-point error;
- D2-like point-to-plane error with a declared normal-estimation protocol;
- symmetric Chamfer RMSE;
- 95th/99th percentile and Hausdorff distance;
- normal consistency, boundary recall, and thin-structure recall.

### Systems

- encode/decode wall time;
- peak CPU/GPU memory;
- output density scalability;
- robustness to permutation, rigid transforms, quantization, and dropped
  constellation points.

## First stop/go gate

Train on procedural geometry and evaluate on held-out parameter ranges.

Proceed if the coordinate-only model:

1. beats farthest-point sampling plus the same decoder at matched bits;
2. degrades smoothly under 10-12 bit coordinate quantization;
3. generalizes to held-out primitive sizes, poses, and combinations; and
4. keeps anchors visibly near the represented surface.

Reframe the idea as learned simplification or hybrid primitive coding if it
requires high-precision off-surface anchors, feature channels, or a category-
specific decoder to work.

## Milestone 2: adaptive variable-rate set codec

Replace independently trained fixed-`K` models with one set-to-set model that:

- accepts variable input cardinality `N`;
- selects `K` per cloud under an explicit rate objective;
- reconstructs a variable cardinality `M` without dropping transmitted points;
- can choose quantized raw pass-through when compression is counterproductive;
- exposes no per-point features at the bottleneck; and
- counts mode, cardinality, precision, bounds, and entropy syntax in the rate.

Begin with a masked point/set transformer and learned token halting. Treat a
conditional diffusion decoder as a later reconstruction-quality ablation, not
as a change to the coordinate-only message.
