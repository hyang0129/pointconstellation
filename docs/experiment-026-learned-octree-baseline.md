# Experiment 026: learned shallow-octree baseline

## Purpose

Experiment 026 tests the most dangerous remaining competitor to the reported
Point Constellation low-rate corridor: a learned entropy model over shallow
octree occupancy. Unlike the Experiment 020 `pcc_geo_cnn_v2` pilot, this
baseline preserves only the geometry selected by octree depth and can avoid
much of G-PCC's fixed syntax overhead.

The selected implementation is
[OctAttention](https://github.com/zb12138/OctAttention), object branch commit
`adb628b29abc4b160f55fe27dd43b0db7b730cac`, under Apache-2.0. VoxelContext-Net
is not used because OctAttention is feasible with a small interface patch; no
evidence currently justifies substituting the fallback.

This document predeclares the protocol. No Experiment 026 cluster result is
part of this change.

## Hypothesis and decision rule

The hypothesis is that learned shallow-octree probabilities remove enough
fixed syntax cost to match or dominate the stabilized constellation near
0.15--0.25 bits per 2,048-point source point. The primary comparison is paired
per-cloud official symmetric D1 RMSE on the common 12-bit grid at matched
actual payload bytes. Official symmetric D2 RMSE is a required co-primary
diagnostic; Chamfer is secondary.

The predeclared reading is strict: **if OctAttention at any depth at most 6
matches the stabilized constellation at matched payload bytes, withdraw the
Point Constellation rate-distortion corridor claim.** Depth 7 is retained to
show the transition out of the target rate region, not to weaken that rule.

The intended payload window is 20--200 bytes. Observed streams outside that
window remain reported; results are not removed or extrapolated to manufacture
overlap.

## Fixed data and rates

The source records are the same deterministic normalized 2,048-point clouds
declared by `configs/experiment_019_stability_modelnet40.json`. Every source is
first quantized once to the global 12-bit grid. OctAttention then maps that
fixed occupancy to each global octree depth in `{4, 5, 6, 7}`. The resulting
occupied voxels, rather than the original floating-point sample or point
ordering, are the codec input.

Evaluation uses Experiment 019 validation and category-OOD records. Retraining
uses exactly its 512 `train` mesh records. Calibration, validation, and OOD
records are not exported to the external training tree. The exporter records
the source PLY hash, unordered q12 occupancy hash, mesh identity, sample seed,
and Experiment 019 manifest SHA-256. It refuses duplicate identities and an
incorrect training count.

Two arms are kept separate throughout manifests, directories, and result rows:

- `pretrained_transfer_mpeg_8i_mvub` is the released object checkpoint. Its
  upstream training content is MPEG 8i and JPEG MVUB, not the Experiment 019
  split. It is a pretrained-transfer diagnostic and cannot establish a fair
  ModelNet40 comparison.
- `retrained_exact_experiment_019_train` uses fresh seeds 7, 17, and 29 and only
  the exact 512 exported training sources. One model per seed is trained jointly
  across depths 4--7. The architecture, cross-entropy objective, Adam learning
  rate `1e-3`, upstream network batch of 32, and context length 1024 are retained;
  the deterministic adapter expresses the budget as 100,000 optimizer steps.

Checkpoint bytes are shared model cost and are reported separately. They are
never added to or hidden inside per-cloud payload bytes. An amortized model-cost
claim requires a separately declared population size.

## Why a patch is required

The released `encoder.py` and `decoder.py` are single-file research scripts
with paths and checkpoint names embedded in source. More importantly, released
decode opens the encoder-generated MAT file to obtain the octree-symbol count,
quantizer, offset, and a copy of the true occupancy sequence used by debug
assertions. That is not an independent bitstream decode and cannot support the
depth sweep as published.

`patches/octattention_lowrate.patch` adds one adapter file to the pinned
checkout. Its file SHA-256 is
`81d081e63061b9cb5b922b2548fb346ab52912c4b4e8f76e1bb37e78961287f2`;
the resulting `git diff --binary HEAD` SHA-256 is
`81d081e63061b9cb5b922b2548fb346ab52912c4b4e8f76e1bb37e78961287f2`.
The runtime harness rejects any other checkout diff.

The adapter makes only protocol and execution changes:

- accepts batch input, stream, reconstruction, checkpoint, and depth paths;
- maps the already quantized q12 occupancy to a declared global shallow grid;
- prefixes the upstream arithmetic payload with a 19-byte header containing
  magic, format version, q12 precision, depth, symbol count, and payload length;
- decodes from that stream and the shared checkpoint without opening an input
  PLY or encoder MAT file; and
- provides deterministic exact-split training and records its source-manifest,
  checkout, seed, budget, and final-checkpoint hashes.

The header is counted in every measured stream. The patch does not add or
change neural layers, probabilities, arithmetic coding, decoded occupancy, or
the released checkpoint. If a cloud does not occupy the declared global depth,
the adapter fails rather than silently switching coordinate systems.

## Metrics and contract checks

The generic `ExternalCodecSpec` adapter writes q12 input PLYs, invokes batch
encode and batch decode as separate subprocesses, measures the actual stream
file length, and records stream and reconstruction SHA-256 values. MPEG
`pc_error` then measures each reconstruction on the 12-bit grid with the source
normals used by Experiment 020. Per-cloud rows also include unique depth-grid
voxels, D1/D2 and Hausdorff values, encode/decode time, checkpoint bytes, commit,
and checkout-diff identity. The harness reconstructs the expected q12 voxel
representatives independently and requires exact unordered occupancy agreement
with decoded output before accepting the run.

The issue 18 diversity check is applied independently to every arm, seed, and
depth. A rate point must contain at least two distinct shallow-grid inputs, at
least two stream hashes, and at least two reconstruction hashes. Constant
payload or constant-output failures invalidate the run rather than appearing
as a low-rate result.

No conclusion should be based only on mean bytes. The paper analysis must show
the per-cloud byte distribution, paired distortion deltas at overlapping bytes,
three-seed retrained uncertainty, validation and category-OOD results, and
qualitative failures. The pretrained arm remains labeled even if it performs
better.

## Reproduction

Prepare the pinned checkout and isolated PyTorch environment on a Linux GPU
host:

```bash
scripts/prepare_octattention.sh --create-env --conda /path/to/conda
```

The stages below dispatch tracked jobs through the existing EmpireAI Jupyter
runner. They are intentionally separate so checkpoint evaluation cannot begin
before all declared retraining seeds finish:

```bash
scripts/launch_octattention_empire.sh export
scripts/launch_octattention_empire.sh train
scripts/launch_octattention_empire.sh evaluate
```

Use `--smoke` for a two-step training plumbing check or a two-cloud-per-split
evaluation. A smoke result is not scientific evidence. To run the evaluator
directly after preparation and retraining:

```bash
POINTCONSTELLATION_CODEC_GPU=0 \
  .venv-train/bin/python -m pointconstellation.octattention_benchmark \
  --config configs/external/octattention_lowrate.json
```

Expected outputs are a run manifest, immutable per-cloud PLY/stream/hash rows,
separate subprocess logs, rate summaries, diversity diagnostics, and official
metrics under `artifacts/local/experiment_026_octattention_lowrate/`. Large
external checkouts, training exports, checkpoints, and result artifacts remain
uncommitted.
