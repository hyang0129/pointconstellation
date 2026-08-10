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

## Current milestone: a falsifiable plane baseline

The repository begins with a dependency-light, deterministic baseline:

1. fit a plane to a point cloud with PCA;
2. project the cloud into the fitted plane;
3. encode its rectangular extent as four 3D corner points;
4. decode those unordered corners into a dense point sample; and
5. report Chamfer RMSE, Hausdorff distance, and plane-fit error.

This is not the final learned codec. It verifies the geometry-only API and
provides a sanity-check that future models must beat.

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

The first learned experiment will compare a coordinate-only constellation
autoencoder against farthest-point sampling, learned simplification, a latent
point model with feature channels, and conventional codecs at matched rates.
The decisive ablations are:

- coordinates only versus coordinates plus latent features;
- floating-point versus quantized/noisy constellation coordinates;
- unrestricted versus surface-proximal anchors;
- rigid-motion equivariant versus ordinary encoder/decoder; and
- fixed `K` versus a rate-distortion objective with variable `K`.

See the [field map](docs/research-landscape.md) and
[experiment plan](docs/experiment-plan.md) for the prior art, novelty boundary,
metrics, datasets, and stop/go criteria.

## Project status

This repository is at research milestone 0. The plane codec is usable; claims
about learned compression performance have not been made. Issues and small,
reproducible experiments are welcome.

## License

[MIT](LICENSE)
