# Experiment 023: equal-protocol feature-codec comparison

## Status

This experiment is predeclared and not yet run at scale. It addresses the
training-protocol confound in the Experiment 019 learned-codec comparison.
Experiment 019 compared a four-epoch, EMA-selected constellation decoder with
the two-epoch raw-final feature codecs from Experiment 018. The resulting
29.90% validation Chamfer RMSE difference therefore does not isolate the
representation.

The feature latent is an explicit non-coordinate-only baseline. Its complete
per-cloud message is an ordered, fixed-width feature vector plus its bitstream
header. It is not evidence for the coordinate-only bottleneck contract.

## Hypothesis and primary gate

The primary estimand is the stabilized constellation's relative validation
source-cloud Chamfer RMSE improvement over the feature codec at 50 bytes:

```text
100 * (feature RMSE - constellation RMSE) / feature RMSE
```

The predeclared gate passes only if the stabilized constellation is at least as
good as the equal-protocol feature codec at 50 bytes and the independent-seed
95% bootstrap confidence interval excludes zero on the positive side. In the
machine-readable output this is `confidence_interval_lower_percent > 0`.

If this gate fails, the representation claim becomes **"competitive with a
byte-matched feature codec"**. Experiment 019's 29.90% result must not be cited
as evidence that coordinates outperform a feature latent. This disposition is
about the representation claim; a confidence interval favoring the feature
codec would require correspondingly stronger wording than "competitive".

The 32-, 86-, and 158-byte cells are secondary rate-curve measurements.
Category OOD and fresh-resampling Chamfer are secondary robustness outcomes.
They do not replace the primary validation gate.

## Equal training and data protocol

The feature codec uses the same stabilization choices as the Experiment 019
decoder arm:

- four training epochs and the same cyclic four-rate curriculum;
- EMA updated after every optimizer step with decay 0.99;
- EMA candidates at epochs 2, 3, and 4;
- checkpoint selection by aggregate source-visible calibration Chamfer RMSE at
  the primary 50-byte rate; and
- no validation, category-OOD, target-resampling, or official-metric input to
  checkpoint selection.

The feature encoder and decoder are jointly covered by EMA. Candidate scoring
temporarily loads an EMA state, hashes and copies it, and restores the raw
training state through the same helper used by the Experiment 019 decoder.
Earliest epoch wins an exact calibration-score tie.

The main configuration reuses the Experiment 019 ModelNet40 manifest, seed,
512 training clouds, 128 calibration clouds, 128 validation clouds, and 32
held-out-category clouds. It trains six independent feature-codec seeds:
7, 17, 29, 41, 53, and 67. Equal numeric seed labels do not pair feature and
coordinate model draws; the two model families are resampled independently.

## Exact rate matching

Both formats count their complete fixed-width headers. Configuration loading
rejects any cell for which
`expected_feature_stream_bytes(latent_dim, feature_bits)` differs from
`expected_stream_bytes(K, coordinate_bits)`.

| Constellation `K` | Coordinate precision | Feature dimension | Feature precision | Stream bytes |
|---:|---:|---:|---:|---:|
| 4 | 12 bits | 20 | 8 bits | 32 |
| 8 | 12 bits | 38 | 8 bits | 50 |
| 16 | 12 bits | 74 | 8 bits | 86 |
| 32 | 12 bits | 146 | 8 bits | 158 |

These are actual deterministic streams, not tensor-size estimates or
entropy-coded rates. Shared encoder/decoder checkpoint sizes are recorded
separately and excluded from the per-cloud rate.

## Statistical comparison

At 50 bytes, every stabilized Experiment 019 decoder-by-refiner cell is
compared with every independent Experiment 023 feature seed on identical cloud
identities. The analysis reuses `_independent_feature_bootstrap`:

- coordinate decoder seeds, coordinate refiner seeds, and feature-codec seeds
  are resampled as independent model factors;
- category and cloud draws are paired between representations; and
- categories are sampled hierarchically before clouds within category.

Before aggregation, the runner checks the manifest hash, partition membership
hashes, data seed, point count, sample counts, batch size, rate curriculum,
primary rate, coordinate precision, epoch count, EMA decay, selection start,
and factorial completeness against the saved Experiment 019 artifact. A
protocol mismatch stops the comparison.

## Commands and artifacts

The fixture smoke run exercises EMA and calibration selection on CPU or MPS:

```bash
.venv-train/bin/python -m pointconstellation.feature_codec_benchmark \
  --config configs/experiment_023_feature_codec_smoke.json \
  --device mps
```

The six-seed run expects the ignored ModelNet40 stability manifest and the
completed Experiment 019 artifacts at the paths declared in the main config:

```bash
.venv-train/bin/python -m pointconstellation.feature_codec_benchmark \
  --config configs/experiment_023_feature_codec_equal_protocol.json \
  --device auto \
  --resume
```

On an EmpireAI login node, first obtain and synchronize a Jupyter allocation as
described in [EmpireAI GPU workflow](empire-ai.md), then dispatch:

```bash
scripts/launch_experiment_023_empire.sh
```

The main output directory is
`artifacts/local/experiment_023_feature_codec_equal_protocol`. Each seed writes
selected encoder and decoder checkpoints, `model/selection.json`, per-cloud
JSONL, and seed metrics. `multiseed_metrics.json` records protocol checks,
independent-seed comparisons, and the predeclared gate. Generated artifacts and
local dataset manifests remain outside Git.
