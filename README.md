# Point Constellation

Point Constellation is a research project for lossy point-cloud geometry
compression through a **geometry-only bottleneck**.

Instead of transmitting a feature vector or attaching learned features to a
sparse set of points, the proposed codec transmits only a much smaller point
set in the same 3D coordinate space:

```text
X in R^(N x 3)  --encoder-->  Z in R^(K x 3)  --decoder-->  X_hat in R^(N x 3)
                                  K << N
```

`Z` is the *constellation*. Its geometry is the entire per-cloud message. A
shared decoder learns how arrangements of constellation points expand into
dense geometry.

For example, a wall sampled by 500 points can be represented by four coplanar
corner points. Those four points carry the plane's position, orientation, and
extent through geometry alone.

> [!IMPORTANT]
> Reducing 500 coordinates to 4 is a representation ratio, not yet a valid
> compression result. A codec must quantize and entropy-code the constellation,
> count all required metadata, and compare rate-distortion curves against
> established codecs.

## What makes the hypothesis strict

The bottleneck permits only an unordered `K x 3` coordinate set. It forbids
per-point features, colors, normals, labels, connectivity, token types,
meaningful array order, and per-cloud decoder weights. Quantization and jitter
are required during learning so the encoder cannot hide an arbitrary feature
vector in imperceptible coordinate perturbations.

The complete rules are in the
[representation contract](docs/representation-contract.md).

## Current implementation

The repository begins with a dependency-light, deterministic baseline:

1. fit a plane to a point cloud with PCA;
2. project the cloud into the fitted plane;
3. encode its rectangular extent as four 3D corner points;
4. decode those unordered corners into a dense point sample; and
5. report Chamfer RMSE, Hausdorff distance, and plane-fit error.

This analytic path is not the final learned codec. It verifies the
geometry-only API and provides a sanity-check that learned models must beat.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pointconstellation.demo --points 500 --noise 0.005
pytest
```

Example output fields:

```json
{
  "input_points": 500,
  "constellation_points": 4,
  "coordinate_ratio": 125.0,
  "chamfer_rmse": "measured at runtime",
  "hausdorff": "measured at runtime"
}
```

## Research direction

The core experiment is explicitly an **ML encoder/decoder**, not an analytic
primitive codec. The final target accepts a variable-size input set, chooses a
variable-size unordered constellation, and produces a variable-size output set.
Compression is optional: when a smaller geometric message is not worthwhile,
the codec should preserve the original coordinates through a raw/pass-through
endpoint instead of forcing a lossy bottleneck.

The current fixed `N -> K -> N` model is a controlled precursor. Its encoder
learns to reduce a dense cloud to a quantized `K x 3` constellation, and its
decoder receives only those coordinates. See the
[adaptive codec target](docs/adaptive-codec.md) for the transformer-first design,
rate-controlled cardinality, preservation rule, and diffusion-decoder ablation.

The first learned experiment compares this coordinate-only constellation
autoencoder against farthest-point sampling with the same decoder and raw
coordinate rate. Later experiments add learned simplification, a latent point
model with feature channels, and conventional codecs at matched coded rates.
The decisive ablations are:

- coordinates only versus coordinates plus latent features;
- floating-point versus quantized/noisy constellation coordinates;
- unrestricted versus surface-proximal anchors;
- rigid-motion equivariant versus ordinary encoder/decoder; and
- fixed `K` versus a rate-distortion objective with variable `K`.

See the [field map](docs/research-landscape.md) and
[experiment plan](docs/experiment-plan.md) for the prior art, novelty boundary,
metrics, datasets, and stop/go criteria.

Implementation is tracked in [Experiment 1](https://github.com/hyang0129/pointconstellation/issues/1).

### Local ML smoke run

The first real encoder/decoder slice can run on Apple MPS, CUDA, or CPU:

```bash
uv venv --python 3.13 .venv-train
uv pip install --python .venv-train/bin/python -e '.[train,dev]'
.venv-train/bin/python -m pointconstellation.train \
  --config configs/experiment_001_smoke.json
```

This smoke configuration uses 256 input points, a 16-point coordinate-only
constellation, and 12-bit training quantization. Outputs go to ignored
`artifacts/local/`; it is a pipeline validation, not a compression benchmark.
The first MPS run and its limitations are recorded in the
[local smoke report](docs/experiment-001-local-smoke.md).

Run the paired learned-versus-FPS gate with:

```bash
.venv-train/bin/python -m pointconstellation.compare \
  --config configs/experiment_001_fps_comparison.json
```

Both models receive the same procedural data order, decoder architecture,
decoder initialization, quantizer, 16-point constellation, and 576-bit raw
coordinate budget. The learned model's validation Chamfer RMSE was 4.29% lower
in the first local run and was lower for all seven procedural families. See the
[matched FPS report](docs/experiment-001-fps-comparison.md) for the result and
its important limitations.

The next fixed-rate control sweeps constellation size and precision:

```bash
.venv-train/bin/python -m pointconstellation.sweep \
  --config configs/experiment_001_rate_sweep.json
```

These independently trained points establish the rate-distortion targets that
the later single adaptive model must match or beat.

The first local sweep found that learned constellations beat FPS at all 12
points, but also exposed a limitation: the learned model was essentially flat
from 96 to 1,152 payload bits. The current global-pooling architecture does not
convert extra anchors into better fidelity. See the
[rate sweep report](docs/experiment-001-rate-sweep.md).

Experiment 002 replaces that bottleneck with relation-aware point and anchor
attention, then trains longer selected-rate curves at `K=4, 16, 32`. It also
evaluates held-out procedural parameter ranges and compares learned anchors
with FPS coordinates feeding the same relation-aware decoder:

```bash
.venv-train/bin/python -m pointconstellation.selected_rate \
  --config configs/experiment_002_relation_aware.json
```

Its quantitative hypothesis and stop/go gate are defined in the
[relation-aware experiment plan](docs/experiment-002-relation-aware-plan.md).
The first local run passed its monotonic-rate gate, but FPS anchors with the
same relation-aware decoder substantially outperformed learned anchors at
K=16 and K=32. See the
[Experiment 002 result](docs/experiment-002-relation-aware-result.md).

Experiment 003 isolates that encoder gap with a learned hard input-subset
selector and a projection-temperature/surface-loss grid:

```bash
.venv-train/bin/python -m pointconstellation.encoder_isolation \
  --config configs/experiment_003_encoder_isolation.json
```

The hard selector produces genuine quantized input subsets but remained 28.12%
worse than FPS on validation. All soft encoders collapsed their anchors. See
the [Experiment 003 result](docs/experiment-003-encoder-isolation-result.md).

Experiment 004 separates decoder learning from encoder selection. One
completion decoder is trained across variable input and constellation sizes,
then frozen while FPS, random, learned progressive subsets, best-of-sampled
subsets, and free coordinates are compared:

```bash
.venv-train/bin/python -m pointconstellation.bottleneck_audit \
  --config configs/experiment_004_frozen_decoder.json
```

The shared decoder produced a clean monotonic FPS rate curve, but the learned
subset remained 41.65% worse than FPS on validation. Per-cloud free coordinates
beat the best sampled subset by 15.60% while moving away from observed samples;
the conditions used different random candidate pools and the current metric
does not establish continuous-surface distance. See the
[Experiment 004 result](docs/experiment-004-frozen-decoder-result.md).

### Three constellation-inference prototypes

The co-adaptation review produced three implemented experiment paths:

```bash
# Competitive semi-amortized refinement against a frozen decoder.
.venv-train/bin/python -m pointconstellation.refiner_experiment \
  --config configs/experiment_005_refiner_smoke.json --device cpu

# Coordinate auto-decoder, noisy neighborhoods, and held-out amortization.
.venv-train/bin/python -m pointconstellation.auto_decoder_experiment \
  --config configs/experiment_006_auto_decoder_smoke.json --device cpu

# Multimodality-gated conditional set diffusion from noisy FPS.
.venv-train/bin/python -m pointconstellation.diffusion_experiment \
  --config configs/experiment_007_set_diffusion_smoke.json --device cpu
```

All three are CPU-runnable and retain a coordinate-only transmitted message.
The first smoke run favors the competitive refiner: its free-coordinate output
improved validation at all tested points, whereas auto-decoder gains were small
and diffusion remained worse than FPS. See the [prototype implementation
report](docs/experiment-005-007-prototypes.md) for architectures, metrics, and
caveats.

The corrected [scaled refiner run](docs/experiment-005-refiner-scale-result.md)
improved the primary `N=256, K=16` validation RMSE from `0.18261` to `0.14073`
with free coordinates and `0.15625` after strict unique projection to input
points. A no-decoder-gradient arm reached `0.17334`, isolating a smaller learned
refinement gain from the larger, legal input-only gradient-feedback gain.

The first [multi-seed benchmark](docs/experiment-013-refiner-multiseed-result.md)
replicated that primary result with three independently trained decoders and
paired statistics. At `N=256, K=16, q=12`, free input-gradient constellations
beat FPS by 23.33% on validation (95% CI 20.44-26.62) and 25.13% on
parameter OOD (20.73-29.50); post-hoc strict projection retained 14.51% and
15.18% gains. This passes the procedural primary-point gate but is not yet an
external-data or actual-rate compression claim.

Five more co-adaptation hypotheses are implemented in Experiments 008-012:
balanced transport, compression homotopy, decoder populations/cross-play,
gradient-free search, and autoregressive pointer selection. Their [fan-out
report](docs/experiment-008-012-fanout.md) records matched contracts, corrected
controls, smoke results, and the decision to keep the competitive refiner as
the leading scale direction.

## EmpireAI GPUs

The repository includes secret-free EmpireAI tooling adapted from HalluLens:
a guarded Jupyter allocation launcher, live SLURM node discovery, remote
Jupyter execution, GPU-aware dispatch, and job tracking. It assumes the local
SSH alias `empire-ai`; keys and passwords stay outside Git.

See the [EmpireAI guide](docs/empire-ai.md). No GPU allocation or training job
is submitted automatically.

## Project status

This repository is in research milestone 1 and is working toward a
theory-backed, comprehensively benchmarked ICLR/NeurIPS-quality paper. The
plane codec, fixed-rate ML baselines, rate sweeps, and refiner benchmarks are
runnable; claims about competitive general-purpose compression performance
have not been made. The [paper benchmark epic](https://github.com/hyang0129/pointconstellation/issues/3)
tracks the remaining evidence.

## License

[MIT](LICENSE)
