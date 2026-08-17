# Experiment 020: official metrics and published learned codec

Status: official-metric slice and published-codec pipeline/rate smoke complete;
fair external retraining and full curves remain open under
[issue #15](https://github.com/hyang0129/pointconstellation/issues/15).

## Decision summary

The stabilized Experiment 019 result survives official MPEG measurement. At
the exact 50-byte `N=2048,K=8,q=12` point, the competitive refiner improves
official validation D1 RMSE by 16.52% and D2 RMSE by 34.65% over FPS decoded by
the same frozen model. Category-OOD improvements are 18.80% and 33.78%.
Every one of the six decoder marginals improves on both metrics and splits.

This passes Experiment 020 Gate A and removes the risk that the earlier Chamfer
gain was an artifact of the internal proxy metric. It does not pass the
published-codec gate: the released external points do not overlap the 50-byte
rate, and a fair retrained full curve has not yet executed.

## Fixed contract

- The evaluation reuses the exact Experiment 019 ModelNet40 manifest, source
  samples, six decoder seeds, three refiner seeds, and sealed calibration
  selections.
- Encoder-side decoder gradients inspect only the source cloud.
- Every constellation is serialized and decoded before reconstruction.
- Every complete stream is 50 bytes, or 0.1953125 bits per input point.
- Official `pc_error` runs on the common 12-bit integer grid using source
  normals.
- FPS is evaluated once per decoder; all three refiner seeds are crossed with
  every decoder.
- Validation and category OOD do not select a model, checkpoint, or method.

## Official D1/D2 result

| Split | Metric | FPS RMSE | Refiner RMSE | Improvement | 95% CI | Decoder wins |
|---|---|---:|---:|---:|---:|---:|
| validation | D1 | 316.738 | 264.414 | 16.52% | 11.43--21.79 | 6/6 |
| validation | D2 | 260.932 | 170.507 | 34.65% | 27.07--40.81 | 6/6 |
| category OOD | D1 | 359.917 | 292.268 | 18.80% | 8.77--25.50 | 6/6 |
| category OOD | D2 | 315.863 | 209.172 | 33.78% | 26.45--41.37 | 6/6 |

RMSE values are in the declared 12-bit integer-grid units. Intervals use a
paired hierarchical category/cloud bootstrap with paired decoder and common
refiner factor resampling. The primary validation gate requires every decoder
marginal to improve and both D1 and D2 lower confidence bounds to exceed zero.
It passes.

All 3,840 per-cloud rows completed in 275.22 seconds on Apple MPS. The exact
stream, lattice, frozen-decoder, sealed-selection, data-identity, and
source-only-gradient checks all pass. Per-cloud rows and generated MPEG working
files remain ignored.

## Published external codec

The first external method is the official
[`pcc_geo_cnn_v2`](https://github.com/mauriceqch/pcc_geo_cnn_v2) implementation
of *Improved Deep Point Cloud Geometry Compression*, pinned to commit
`b7a4ae2a548ad3c44a04af139dd77d804cf3a6fa`. It is the closest first comparison
because its released training workflow uses ModelNet40, it emits complete gzip
bitstreams, and it reports official D1/D2.

The repository now includes:

- a black-box subprocess adapter that verifies the upstream commit, executes
  compression and independent decompression without a shell, counts the actual
  stream, hashes the stream and decoded PLY, accepts ASCII or binary PLY,
  captures logs, and records the environment;
- a pinned five-lambda `c4-ws` protocol on the upstream 512³ ModelNet grid,
  partitioned into its native 64³ occupancy blocks, with reconstructions
  measured on the common 12-bit metric grid;
- a guarded preparation script for the legacy Python 3.7, TensorFlow 1.15, and
  tensorflow-compression 1.3 environment; and
- a resumable published-codec runner that evaluates actual streams with
  Chamfer and official D1/D2.

The authors' 5,523,313,425-byte Zenodo release was downloaded on EmpireAI and
verified against its declared MD5
`dc930e34c2ad1f8a360b80a313df9526`. Third-party code, models, streams, and
environments remain outside Git. The reproducible environment records Python
3.7, TensorFlow GPU 1.15.5, TensorFlow Compression 1.3, Protobuf 3.20.3, CUDA
10.0, cuDNN 7.6.5, and successful H200 visibility.

An initial q=12 invocation was deliberately stopped after exposing a protocol
mismatch: one 2,048-point validation cloud fragmented into 1,333 mostly sparse
64³ blocks, whereas the released model was trained on individual 64³ occupancy
blocks. The external arm therefore uses the upstream-native q=9 ModelNet input
grid and level-3 octree partition. Its precision loss remains part of its
measured distortion; Point Constellation continues to use its declared q=12
stream, and `pc_error` measures both reconstructions on the same q=12
evaluation grid.

### Released-checkpoint rate smoke

All five released `c4-ws` checkpoints produced nonempty gzip streams for the
same exact validation cloud (`airplane_0641`, 2,048 source points):

| Lambda | Stream bytes | Bits/source point | Multiple of 50 bytes |
|---:|---:|---:|---:|
| 3.00e-04 | 8,817 | 34.441 | 176.34x |
| 1.00e-04 | 6,385 | 24.941 | 127.70x |
| 5.00e-05 | 4,884 | 19.078 | 97.68x |
| 2.00e-05 | 2,918 | 11.398 | 58.36x |
| 1.00e-05 | 1,751 | 6.840 | 35.02x |

For the middle checkpoint, independent decompression of the 6,385-byte stream
reproduced the encoder-side PLY byte-for-byte (SHA-256
`0770761bb1a73538f21a0ce0c366fea8e439353e8482b87352ccc777880c7e8f`).
It decoded 17,444 points and measured D1 MSE 280.80 and D2 MSE 49.84 on the
common q=12 grid. These quality values are only pipeline diagnostics because
of the released-checkpoint leakage caveat below.

The decisive result is the rate scale: even the smallest released point is
35x larger than Point Constellation's primary stream. There is no overlapping
rate interval, so no BD-rate or superiority claim is valid. The full harness
batches clouds per checkpoint to amortize the legacy model startup, then runs
an independent decoder and official metrics on every reconstruction.

### Leakage constraint

The released checkpoints are useful for an environment and bitstream smoke,
but not automatically a fair primary ModelNet40 result. The upstream dataset
preparation selects ModelNet40 files by a recursive glob and does not establish
that its training set excludes this repository's official-test validation and
held-out-category records. A pretrained comparison could therefore contain
test overlap.

The paper comparison must retrain the published architecture on the exact 512
Experiment 019 training meshes, or prove record-level disjointness from the
released training set. Until then, released-checkpoint results are labeled
pipeline diagnostics rather than evidence of superiority.

## Reproduction

Run or resume the official metric slice with:

```bash
.venv-train/bin/python -m pointconstellation.official_stability \
  --config configs/experiment_020_official_stability.json \
  --device mps
```

Prepare the external checkout, release, and isolated environment on Linux:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/prepare_pcc_geo_cnn_v2.sh \
  --conda /path/to/conda \
  --create-env \
  --download-artifacts
```

After the exact training partition and checkpoints are available, run:

```bash
POINTCONSTELLATION_CODEC_GPU=0 \
  .venv-train/bin/python -m pointconstellation.published_codec_benchmark \
  --config configs/experiment_020_pcc_geo_cnn_v2.json
```

## Next decision

### Exact-source retraining protocol

The fair retraining slice is now fixed and executing. A deterministic exporter
materializes the exact 2,048-point source sample for each of the 512 Experiment
019 training meshes and 128 calibration meshes. It does not export validation
or category-OOD records. The resulting 10,424,358-byte external artifact has
SHA-256
`f844433174b3843102adcc1b5de6417a9d6d748be94a24156b99843710a554aa`;
its machine-readable training manifest has SHA-256
`4808da615b332599a8f57c5d8f57de1ea9a5305d02f58920fea024b75d783ddb`.
Regenerating the archive produces the same bytes.

Two arms keep the published `c3p` network, entropy model, focal/rate objective,
optimizer, and 64³ network input unchanged:

- `native_q9_oct3` uses the upstream-aligned 512³ grid and level-three octree;
  the exact sources produce 53,127 occupied training/calibration blocks;
- `low_rate_q6_global` quantizes each entire object to one 64³ block and trains
  five aggressive lambdas from `1e-5` through `1e-7`. This is explicitly a
  low-rate protocol adaptation, not the paper's native ModelNet operating
  point.

The upstream container code does not handle octree level zero despite its
format supporting a zero-length octree. The low-rate evaluator therefore uses
a declared compatibility patch that returns the sole block unchanged during
partition/departition and converts NumPy diagnostic values to JSON-compatible
Python floats. Its Git binary-diff SHA-256 is
`ffba367053a7037ce7b19dc3fea298e412f17970bcd0bd7e8f11152c760bee26`.
The adapter rejects any clean or patched checkout whose exact diff identity
does not match its manifest. The patch does not change neural layers, learned
parameters, entropy tensors, thresholds, reconstruction, or the serialized
block payload.

As a feasibility diagnostic, the released `1e-5` checkpoint encoded the same
`airplane_0641` source as one q6 block into 111 actual bytes (0.4336 bits per
source point, 2.22x the Point Constellation stream). Independent decompression
matched the encoder reconstruction byte-for-byte and produced 2,095 points.
On the common q12 metric grid it measured D1 MSE 12,054.75 and D2 MSE 5,060.20.
This released-checkpoint row retains the leakage caveat and is not a fair
comparison, but it establishes that removing block fragmentation can bring the
published architecture close enough for an overlap search.

The original CUDA 10 stack also failed on H200 during TensorFlow Compression's
small batched entropy matrix gradient. The declared training-only portability
patch places those matrix operations and gradients on CPU while keeping 3D
convolutions and the remaining optimization graph on GPU. A 500-step sealed
checkpoint completed successfully before launching the five-point pilot.

The first 5,000-step runs are rate-feasibility pilots. If at least three q6
points approach the 50-byte interval, the declared 100,000-step budgets will
be completed before distortion comparison. A non-overlap after q6 global
coding would establish a stronger framing/entropy floor and direct the next
work toward a new entropy model rather than more training.

Gate B still requires at least three genuinely overlapping actual-rate points;
otherwise the nearest observed rate gap is reported without extrapolation.

AnyPcc remains the next modern comparison on one of its native dense/LiDAR
benchmarks. Its CUDA, TorchSparse, DeepCABAC, per-instance adaptation, and
dataset protocol should not be collapsed into a mismatched ModelNet invocation.
