# Experiment 014: standardized low-rate toy benchmark

## Purpose

Experiment 014 moves the procedural benchmark closer to the low-rate object
protocols used by learned point-cloud compression papers without pretending
that procedural shapes are ShapeNet, ModelNet40, or an MPEG common test set.
It is a **protocol-aligned procedural proxy** and a rehearsal for the real-data
and codec comparisons tracked in issues
[#6](https://github.com/hyang0129/pointconstellation/issues/6),
[#7](https://github.com/hyang0129/pointconstellation/issues/7), and
[#9](https://github.com/hyang0129/pointconstellation/issues/9).

The profile deliberately remains runnable on a MacBook. It uses 2,048 input
and reconstruction points, the scale commonly used by low-rate object-cloud
work, while retaining the repository's deterministic seven-family procedural
dataset.

## Frozen protocol

- Input and reconstruction cardinality: `N=2048`.
- Coordinate domain: each normalized cloud lies in `[-1, 1]^3`.
- Coordinate precision: 12 bits per axis.
- Constellation sizes: `K in {4, 8, 16, 32, 64, 128}`.
- Methods: FPS, free input-gradient refinement, and post-hoc strict unique
  projection.
- Message: unordered `K x 3` coordinates only; no point features, order,
  normals, labels, or per-cloud weights reach the decoder.
- Dataset: all seven procedural families, once per validation and parameter-OOD
  split. The output records SHA-256 manifests.
- Training: one deliberately shallow epoch over 42 clouds. Each of the six
  rate points receives seven optimizer updates.

The benchmark serializes every message before decoding. The fixed-width stream
contains a 14-byte versioned header with mode, coordinate precision, normalized
domain, `K`, and requested output cardinality, followed by canonically ordered
bit-packed lattice coordinates. The neural decoder consumes coordinates parsed
back from those bytes.

| K | Payload bits | Nominal payload bpp | Stream bytes | Actual stream bpp |
|---:|---:|---:|---:|---:|
| 4 | 144 | 0.07031 | 32 | 0.12500 |
| 8 | 288 | 0.14062 | 50 | 0.19531 |
| 16 | 576 | 0.28125 | 86 | 0.33594 |
| 32 | 1,152 | 0.56250 | 158 | 0.61719 |
| 64 | 2,304 | 1.12500 | 302 | 1.17969 |
| 128 | 4,608 | 2.25000 | 590 | 2.30469 |

This exposes an important low-rate fact hidden by nominal `3Kq`: the fixed
header almost doubles the rate at `K=4`. Entropy coding is intentionally absent
until fixed-width accounting and round trips are trustworthy.

## Metrics

Evaluation is per cloud and memory bounded. Both the metrics and the refiner's
decoder-gradient signal use exact chunked distances and never materialize a
full `2048 x 2048` distance tensor. The machine-readable output includes:

- symmetric Chamfer MSE and RMSE;
- D1-like maximum directed point-to-point MSE and normalized PSNR;
- D2-like maximum directed point-to-plane MSE using analytic target normals;
- 95th and 99th percentile Euclidean error and Hausdorff distance;
- deterministic sliced-Wasserstein RMS as a cheap EMD proxy;
- encode/decode time, peak process RSS, parameter counts, checkpoint sizes;
- per-cloud rows, data manifests, rate monotonicity, and environment metadata.

The D1/D2 PSNR values use a declared peak distance of two in the normalized
domain. They are not official MPEG `pc_error` results. Sliced Wasserstein is not
exact EMD. Those names retain the `_proxy` suffix in the JSON output.

## MacBook MPS result

Run date: 2026-08-13. Device: Apple MPS. Seed: 1407. The table reports Chamfer
RMSE averaged over seven clouds per split.

| Split | K | Actual bpp | FPS | Free | Free vs FPS | Strict projection | Strict vs FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 4 | 0.12500 | 0.46336 | 0.42553 | +8.17% | 0.43263 | +6.63% |
| Validation | 8 | 0.19531 | 0.36020 | 0.33697 | +6.45% | 0.34230 | +4.97% |
| Validation | 16 | 0.33594 | 0.29929 | 0.28935 | +3.32% | 0.29140 | +2.63% |
| Validation | 32 | 0.61719 | 0.26542 | 0.25876 | +2.51% | 0.25956 | +2.21% |
| Validation | 64 | 1.17969 | 0.24171 | 0.23763 | +1.69% | 0.23769 | +1.66% |
| Validation | 128 | 2.30469 | 0.21935 | 0.21703 | +1.05% | 0.21656 | +1.27% |
| Parameter OOD | 4 | 0.12500 | 0.38092 | 0.35304 | +7.32% | 0.35994 | +5.51% |
| Parameter OOD | 8 | 0.19531 | 0.31308 | 0.29148 | +6.90% | 0.29486 | +5.82% |
| Parameter OOD | 16 | 0.33594 | 0.25228 | 0.24389 | +3.33% | 0.24577 | +2.58% |
| Parameter OOD | 32 | 0.61719 | 0.22091 | 0.21659 | +1.96% | 0.21748 | +1.55% |
| Parameter OOD | 64 | 1.17969 | 0.19967 | 0.19726 | +1.21% | 0.19719 | +1.24% |
| Parameter OOD | 128 | 2.30469 | 0.18023 | 0.17951 | +0.40% | 0.17876 | +0.82% |

All six method/split curves improved monotonically with increasing actual rate.
The free refiner beat matched-decoder FPS at every point; its advantage was
largest at the lowest rates and narrowed as `K` grew. Strict post-hoc projection
also beat FPS at every point in this run. This is useful evidence that the
Experiment 005 mechanism survives a paper-shaped point count and rate grid,
but it is not statistical replication or a state-of-the-art codec result.

Training took 8.07 seconds and bitstream-level evaluation took 13.81 seconds.
Peak process RSS was 469.2 MiB. The decoder has 81,827 parameters and the
refiner 14,211; their checkpoints are 338,182 and 67,212 bytes respectively.

## Reproduction

Install the training extras, then run either profile:

```bash
# Seconds on CPU; validates contracts and output shape only.
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_014_standardized_smoke.json --device cpu

# Full six-rate, seven-family laptop profile.
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_014_standardized_macbook.json --device mps
```

Outputs are written under `artifacts/local/` and include model checkpoints,
`benchmark_metrics.json`, and line-delimited per-cloud records. Add `--resume`
to reuse checkpoints only when their stored model configuration matches exactly.

## What this resolves and what remains

Resolved:

- the same-space message has a deterministic byte representation;
- rate includes metadata and byte padding rather than only `3Kq`;
- the toy task now uses a 2,048-point, six-rate protocol and paper-style metric
  schema;
- the full local evaluation is bounded-memory and MacBook-runnable; and
- the scaled curve preserves the low-rate gain over matched-decoder FPS.

Not resolved:

- only one training seed and seven clouds per split were used;
- the decoder/refiner are deliberately undertrained;
- no real-data generalization is measured;
- no entropy model, G-PCC stream, learned codec, BD-rate, or official metric is
  present; and
- shared model delivery and amortization are reported separately but are not
  folded into the per-cloud rate.

The next step is to keep this exact output schema while replacing the
procedural adapter with the ShapeNetCore pilot and adding overlapping G-PCC and
learned-codec rate points. Official software should replace the proxy D1/D2
metrics for any external state-of-the-art claim.
