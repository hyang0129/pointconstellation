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
primitive codec. The encoder learns to reduce a dense cloud to an unordered,
quantized `K x 3` constellation. The decoder receives only those coordinates
and learns to reconstruct dense geometry. The analytic plane implementation is
only a contract test and sanity baseline.

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

## EmpireAI GPUs

The repository includes secret-free EmpireAI tooling adapted from HalluLens:
a guarded Jupyter allocation launcher, live SLURM node discovery, remote
Jupyter execution, GPU-aware dispatch, and job tracking. It assumes the local
SSH alias `empire-ai`; keys and passwords stay outside Git.

See the [EmpireAI guide](docs/empire-ai.md). No GPU allocation or training job
is submitted automatically.

## Project status

This repository is entering research milestone 1. The plane codec and local ML
experiment are usable; claims about competitive compression performance have
not been made. Issues and small, reproducible experiments are welcome.

## License

[MIT](LICENSE)
