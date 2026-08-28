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

### Regenerate figures

Rebuild the machine-readable registry, then regenerate the registry-backed
selection, rate-distortion, utility, objective, representation, and BD-rate
figures and tables:

```bash
.venv-train/bin/python -m pointconstellation.benchmark_registry --rebuild
.venv-train/bin/python scripts/figures/fig1_selection_baselines.py
.venv-train/bin/python scripts/figures/fig_rd_positioning.py
.venv-train/bin/python scripts/figures/table1_headline.py
.venv-train/bin/python scripts/figures/table_bd_rate.py
.venv-train/bin/python scripts/figures/fig_rate_utility.py
.venv-train/bin/python scripts/figures/fig_objective_pareto.py
.venv-train/bin/python scripts/figures/table_representation.py
```

The Track B outputs use actual serialized bytes, preserve representation,
objective, and evaluation-regime labels from the source artifacts, and report
missing utility metrics explicitly rather than filling them from another arm.

The RD figure contains only measured registry points. Hollow markers identify
dominated measurements, and bands resample model seeds and paired clouds. The
BD table prints `insufficient overlap` unless both curves have at least four
measured points and a common integration interval; it does not extrapolate.

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

Experiment 014 standardizes the toy task around a 2,048-point low-rate object
protocol and a real fixed-width coordinate stream:

```bash
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_014_standardized_macbook.json --device mps
```

The six-rate, seven-family MacBook run finished in about 22 seconds and used
469 MiB peak RSS. It records actual bpp, per-cloud D1/D2-style proxies, tail
error, sliced-Wasserstein, runtime, model size, manifests, and monotonicity.
The refiner beat matched-decoder FPS at all six rates in both procedural splits,
with the largest gain at the lowest rates. See the [Experiment 014
report](docs/experiment-014-standardized-toy-benchmark.md). This is a
protocol-aligned procedural proxy, not a ShapeNet, MPEG, or SOTA codec claim.

Experiment 015 adds manifest-backed mesh surfaces, independent fresh-surface
resampling, actual MPEG G-PCC/TMC13 streams, and official `pc_error` D1/D2:

```bash
bash scripts/build_mpeg_tools.sh
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_015_mesh_fixture_macbook.json --device mps
```

The checked-in fixture validates eight neural and seven overlapping G-PCC rate
points in under a minute on a MacBook. Its [external-pilot
report](docs/experiment-015-external-mesh-gpcc-pilot.md) documents the result
and the local ShapeNetCore manifest workflow. The fixture is infrastructure
evidence, not a real-data or SOTA claim; the licensed ShapeNetCore run remains
the next decisive step.

While ShapeNetCore access is pending, Experiment 016 runs the same protocol on
the official ModelNet40 train/test distribution. Its one-seed, 40-category
pilot found a 14--15% free-refiner gain over matched-decoder FPS at
0.125--0.195 bpp on both validation and category OOD, but G-PCC overtook the
constellation curve around 0.21 bpp. See the [ModelNet40 pilot
report](docs/experiment-016-modelnet40-pilot.md). This supports a three-seed
low-rate scale run, not a general codec or SOTA claim.

Experiments 017-018 complete that three-seed scale slice on 672 selected
ModelNet40 meshes and add an exactly byte-matched learned feature-latent codec.
At the primary K=8 point, the refiner beats matched-decoder FPS by 16.13% on
validation (95% CI 13.49-19.85) and 18.19% on held-out categories
(13.37-24.59). It does not yet beat the feature codec reliably: the aggregate
validation effect is -6.97% (-29.23-10.78), driven by one weak constellation
seed. Official validation D2 nevertheless favors the constellation by 32.99%
(12.03-54.99). The [scale report](docs/experiment-017-018-modelnet40-scale.md)
documents the rates, G-PCC crossover, runtime, negative learned-codec result,
and external codec replication path.

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

Point Constellation exclusively reserves the `86xx` Jupyter-port namespace and
allows up to six logical allocations. Allocation identity is `hostname-port`,
so shared physical GPU hosts do not merge workstreams. Compute must run through
SLURM/Jupyter rather than direct GPU-node SSH.

See the [EmpireAI guide](docs/empire-ai.md). No GPU allocation or training job
is submitted automatically.

## Project status

This repository is in research milestone 1 and is working toward a
theory-backed, comprehensively benchmarked ICLR/NeurIPS-quality paper. The
plane codec, fixed-rate ML baselines, rate sweeps, and refiner benchmarks are
runnable. [Experiment 019](docs/experiment-019-stability.md) reports a stable
six-decoder by three-refiner ModelNet40 result at one 50-byte operating point,
including a positive internal learned-feature-codec comparison.
[Experiment 020](docs/experiment-020-official-and-published-codec.md) confirms
that result with official D1/D2 and adds a pinned complete-codec harness for
`pcc_geo_cnn_v2` on its native ModelNet occupancy grid. Claims about competitive
general-purpose compression performance have not been made: the executed
released-checkpoint smoke is at least 35x above the 50-byte rate, while the fair
retrained curve and common-test-condition comparisons remain open. The
[Experiment 034 placement diagnostic](docs/experiment-034-placement-analysis.md)
adds reproducible category-consistency, occupancy-entropy, mesh-proxy, and
headless qualitative analyses for the coordinate messages; its full ModelNet40
run remains pending. The
[paper benchmark epic](https://github.com/hyang0129/pointconstellation/issues/3)
tracks the remaining evidence.

## License

[MIT](LICENSE)
