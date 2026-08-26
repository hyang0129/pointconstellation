# Experiment 027: Draco-on-subsets and PCGCv2 low-rate baselines

Status: implementation and smoke-test plumbing complete; benchmark and cluster
runs have not been executed.

## Question and decision rule

This experiment separates two explanations for low-rate performance:

1. whether FPS or source-only Adam selects unusually informative coordinates;
2. whether the frozen Point Constellation decoder, rather than selection, is
   responsible for reconstructing a useful surface from those coordinates.

Draco is the conventional coordinate codec control. PCGCv2 is the published
learned sparse-convolution control. The primary rate field is the size of the
complete emitted payload in bytes. Model checkpoints are reported separately
as `model_bytes`; they are not silently added to, or omitted from, per-cloud
rate.

Continue to a full comparison only if independently decoded streams provide at
least five overlapping measured rate points with the Point Constellation curve.
Do not interpolate across a rate gap. A PCGCv2 rate point is invalid if any
cloud decodes empty, if all streams are identical, or if fewer than 90% of its
per-cloud reconstruction hashes are unique. A failed or collapsed point is a
negative feasibility result, not infinite or imputed distortion.

## Common protocol

- Reuse the exact Experiment 019 ModelNet40 manifest, source samples,
  validation/category-OOD partitions, stabilized decoder checkpoints, and
  source normals.
- Adam coordinates must be selected using only source-visible scores. Record
  the Adam budget and decoder seed in the calling benchmark artifact.
- The external decoder starts in a separate process and receives only the
  emitted stream, its shared deployment model where applicable, and declared
  codec configuration.
- Report actual complete payload bytes, bits per 2,048-point source point,
  encode/decode time, binary or checkout hashes, and shared model bytes.
- Measure D1 and D2 with the official MPEG `pc_error` executable on the common
  12-bit metric grid. Draco's `-qp` is its own bounding-box quantizer and is not
  described as the common 12-bit grid.
- Registry-facing rows contain the dataset `family`, `payload_bytes`,
  `stream_bytes` as a compatibility alias, `model_bytes`, stream and
  reconstruction hashes, and the official metrics.

No encoder feature, target-only information, point order, normal, label,
primitive ID, or per-instance weight enters either learned message. Normals are
used only by the post-decode D2 evaluator.

## Draco arms

The fixed K=8 arms in `configs/experiment_027_draco.json` are:

| Codec input | Draco position bits | Reconstruction |
|---|---|---|
| centroid-start FPS subset | 8, 10, 12 | decoded points directly |
| centroid-start FPS subset | 8, 10, 12 | existing frozen decoder |
| source-only Adam-selected constellation | 8, 10, 12 | decoded points directly |
| source-only Adam-selected constellation | 8, 10, 12 | existing frozen decoder |
| full 2,048-point source | 3, 4, 5, 6 | decoded points directly |

The direct subset rows isolate selection quality without the learned decoder.
The matched frozen-decoder rows isolate whether Draco coding preserves enough
of the selected message for the existing decoder. They remain separately
labeled from fixed-width Point Constellation streams and from strict learned
subsets.

`pointconstellation.codecs.draco.run_draco` forces `-point_cloud`, `-qp`, and
`-cl`, hashes both executables, rejects stale output directories, counts the
actual `.drc` file, and invokes `draco_decoder` independently. The adapter does
not count an input PLY or encoder-side reconstruction as payload.

The build script pins Draco 1.5.7 to commit
`8786740086a9f4d83f44aa83badfbea4dce7a1b5`. It forces an arm64 build on Apple
ARM, checks the Mach-O architecture, performs a four-point command-line
round-trip, and writes executable hashes to an ignored build manifest:

```bash
scripts/build_draco.sh
```

Copy the generated encoder and decoder SHA-256 values into the local run config
before producing paper results. The checked config leaves them null because
build products are platform-specific; every result row still records the
actual hashes.

## PCGCv2 low-rate harness

The upstream repository is pinned to commit
`88ff2a18b1b3cac89eef66997cc4e8bcf4fb0420`. The checked harness evaluates the
released `r3_0.10bpp.pth` checkpoint at input voxel precisions 6, 7, and 8 and
at scaling factors 0.25, 0.5, and 1.0. Scaling-factor rows are protocol
adaptations for overlap search, not claimed released paper operating points.
The released model was trained for dense MPEG content, not the exact ModelNet40
split, so its rows are diagnostic until record-level disjointness is proved.

PCGCv2 normally writes four files: losslessly coded bottleneck coordinates,
feature entropy payload, feature header, and point-count metadata. The adapter
frames all four, including component lengths and a stream version, into one
deterministic stream. The complete framed size is `payload_bytes`. Independent
decode unpacks that stream in a new process; no encoder-side tensor or file is
used. The checkpoint file size is `model_bytes`.

Prepare the checkout and released checkpoint, then verify an existing isolated
CUDA/MinkowskiEngine environment on an allocated EmpireAI GPU:

```bash
scripts/prepare_pcgcv2.sh \
  --python /path/to/pcgcv2-environment/bin/python
```

The upstream requirements are Python 3.7/3.8, PyTorch 1.7/1.8, CUDA 10.2/11.0,
MinkowskiEngine 0.5+, torchac 0.9.3, and TMC13 v12. Environment creation is not
automated because CUDA/compiler compatibility is cluster-specific. The prepare
script records Python, package versions, GPU identity, and `pip freeze`; it
fails when CUDA is not visible.

## Exact-split retraining

`configs/experiment_027_pcgcv2_retrain.json` declares q6, q7, and q8 arms with
the upstream architecture and optimizer. The exporter materializes only the
fixed Experiment 019 training and calibration source samples. It rejects any
validation/OOD identity, records every PLY hash, and produces a machine-readable
split manifest:

```bash
.venv-train/bin/python -m pointconstellation.pcgcv2_training \
  --config configs/experiment_027_pcgcv2_retrain.json export
```

The external training adapter uses explicit train and calibration directories;
it does not reproduce the upstream script's implicit sorted 90/10 split and it
never receives validation or category-OOD files. Each final `.pth` checkpoint
is hashed and counted as deployment model bytes.

After obtaining and discovering an EmpireAI Jupyter allocation, launch one arm
through the existing tracked dispatcher:

```bash
scripts/launch_pcgcv2_empire.sh q6_beta4
scripts/launch_pcgcv2_empire.sh q7_beta4 hostname-port
```

These commands launch training and do not cancel or replace cluster jobs. A
cluster run is outside this change. Before scaling beyond the three declared
points, run the complete validation/OOD streams and apply the diversity gate.

## Smoke verification

Local tests use fake Draco and PCGCv2 executables to verify command rendering,
separate encode/decode processes, exact byte accounting, four-component stream
round trips, checkpoint-size accounting, q6--q8 config parsing, deterministic
FPS, coordinate-only training quantization, and constant-output rejection. The
real Draco fixture test skips when `draco_encoder` or `draco_decoder` is absent.

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python -m pytest -q
/Users/hong/code/pointconstellation/.venv-train/bin/python -m ruff format src tests
/Users/hong/code/pointconstellation/.venv-train/bin/python -m ruff check src tests
```

No Draco or PCGCv2 rate-distortion result is reported in this document. That
requires the external binaries/checkpoints, sealed Experiment 019/022 inputs,
and an allocated GPU for PCGCv2.
