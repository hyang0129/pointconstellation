# Header-normalized rate accounting

This note re-summarizes the existing Experiment 017 seed-7 artifacts. It is an
accounting correction, not a new training or codec run. The validation split
has 128 ModelNet40 clouds of 2,048 source points each.

## Definitions

`actual_stream_bpp` remains the complete serialized file size in bits divided
by the source point count. `payload_bpp` uses the byte-aligned payload size, so
padding in the final payload byte is still counted.

For the constellation stream, the complete 14-byte `PCON` header is excluded
from payload-only rate. For TMC13 v23, the parser reads the stream's one-byte
type, four-byte big-endian length, and declared value for every TLV unit. SPS
and GPS byte counts include their TLV framing. `slice_header_bytes` is the
five-byte geometry-brick TLV framing, and the geometry-brick value is counted
as payload. This is conservative: syntax inside the geometry-brick value is
not estimated or subtracted.

The optional sequence-amortized G-PCC accounting for a parameter-set period
of `n` clouds is

```text
total - sps - gps + (sps + gps) / n
```

It is an accounting point only. The resulting per-cloud byte count is not an
independently decodable stream. Set `amortize_parameter_sets_over` in a
standardized benchmark's `gpcc` configuration to emit this accounting rate in
new benchmark rows.

## Experiment 017 seed-7 validation rates

The G-PCC table includes the measured points whose mean full-stream rate is
approximately 0.18 through 0.45 bpp. Values are means over clouds; consequently
mean byte counts need not be integers. Chamfer RMSE is included only to locate
the rates on the existing curve.

| Stream | Rate point | Mean header bytes | Full bpp | Payload bpp | Mean Chamfer RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| G-PCC | `octree_s1_4096` | 37.000 | 0.1797 | 0.0352 | 1.1684 |
| G-PCC | `octree_s1_1536` | 37.000 | 0.1873 | 0.0427 | 0.3363 |
| G-PCC | `octree_s1_1280` | 37.000 | 0.1891 | 0.0446 | 0.2808 |
| G-PCC | `octree_s1_2048` | 37.000 | 0.1911 | 0.0466 | 0.4127 |
| G-PCC | `octree_s1_1024` | 37.000 | 0.2125 | 0.0680 | 0.2151 |
| G-PCC | `octree_s1_768` | 37.000 | 0.2139 | 0.0694 | 0.1627 |
| G-PCC | `octree_s1_640` | 37.000 | 0.2394 | 0.0948 | 0.1344 |
| G-PCC | `octree_s1_512` | 37.133 | 0.2729 | 0.1278 | 0.1058 |
| G-PCC | `octree_s1_256` | 37.055 | 0.4463 | 0.3015 | 0.0548 |
| Constellation, free | `K=4` | 14.000 | 0.1250 | 0.0703 | 0.1612 |
| Constellation, free | `K=8` | 14.000 | 0.1953 | 0.1406 | 0.1445 |
| Constellation, free | `K=16` | 14.000 | 0.3359 | 0.2812 | 0.1282 |
| Constellation, free | `K=32` | 14.000 | 0.6172 | 0.5625 | 0.1166 |

The apparent constellation advantage around 0.19--0.25 full-stream bpp does
not survive payload-only accounting. At full-stream rate, `K=8` has 0.1953 bpp
and 0.1445 mean Chamfer RMSE, while the measured G-PCC points at or below that
rate have higher distortion. After removing the outer headers, the G-PCC
`octree_s1_512` point has both lower rate (0.1278 versus 0.1406 payload bpp) and
lower distortion (0.1058 versus 0.1445). This is a statement about these
measured validation points, not a general codec claim.

## Optional entropy-stream diagnostic

The declared constellation stream remains mode 0: a lexicographically sorted,
fixed-width lattice payload. Mode 1 is an exactly decodable diagnostic over the
same coordinates. It stores the first sorted point at full precision, zigzag
maps subsequent signed deltas, chooses the Rice parameter that minimizes coded
length over the stream, and signals that parameter in one byte. Both modes use
the same 14-byte header. New Experiment 019 and official Experiment 020 rows
report mode 1 as `entropy_stream_bytes` and `entropy_bpp` alongside the unchanged
mode-0 fields. `entropy_bound_bytes` is an oracle per-axis order-0 bound that
includes the header, parameter byte, and full-precision first point, but does
not include the cost of communicating the empirical delta distributions. It is
therefore not a decodable rate.

On the 128-cloud Experiment 019 validation split at `K=8`, `q=12`, the sealed
stabilized refiner factorial contributes 2,304 streams. The fixed stream is
50 bytes (0.1953 bpp) per cloud. The entropy variant averaged 51.888 bytes
(0.2027 bpp; range 43--54), an expansion of 1.888 bytes or 3.776%. The matched
768 FPS rows averaged 52.117 bytes (0.2036 bpp; range 45--54), an expansion of
4.234%. The refiner oracle bound averaged 26.852 bytes (0.1049 bpp), leaving a
25.036-byte mean gap between the implemented Rice stream and the optimistic
bound. Thus this exact mode-1 coder realizes no coding gain on these messages;
the oracle result only identifies distribution-modeling headroom and is not a
compression result.

These values were regenerated from the sealed Experiment 019 models, using the
Experiment 020 official rows only to select the validation messages. The source
hashes were:

- `official_per_cloud.jsonl`: `03694b58ff00ff8c55fa9ce4adb4090afe9588e805cb4e76ba6ca2898b1c9b15`
- `experiment_019_stability_modelnet40.json`: `1dba8d5b6e0533f6da6b6ce34d6837b0dbc0c22b6a1bda388b952c9203f08683`

Reproduce the inference-only resummary without modifying either experiment:

```bash
python scripts/resummarize_entropy_headroom.py \
  --config configs/experiment_020_official_stability.json \
  --official-rows \
    artifacts/local/experiment_020_official_stability/official_per_cloud.jsonl \
  --split validation \
    --output /tmp/experiment_019_validation_entropy_headroom.json
```

## Learned mode-2 stream

Mode 2 preserves the mode-0 coordinate contract and 14-byte `PCON` header. It
codes the same lexicographically sorted lattice with a 32-bit integer arithmetic
coder and appends a four-byte CRC-32 integrity check. The decoder also
re-encodes the decoded lattice and requires the canonical bytes, so truncation,
trailing bytes, a mismatched shared model, and non-canonical arithmetic streams
are rejected. The CRC and any padding are counted in `learned_stream_bytes`.
The declared paper stream remains mode 0.

Two shared-model candidates are implemented:

- `octree` traverses the `q`-level lattice octree, codes child-occupancy bits
  with training-seeded and within-stream adaptive integer contexts, and handles
  duplicate lattice points through exact child-count allocation;
- `autoregressive` uses a small fixed-point MLP conditioned on previously coded
  sorted coordinates. It codes each predicted residual with a stored integer
  discretized-logistic probability table.

Both candidates are fitted from regenerated `split=train` constellations. Their
integer arrays are shared decoder state, are excluded from per-cloud bytes, and
are reported separately as both uncompressed parameter bytes and actual
serialized model bytes. Candidate selection uses validation bytes. The script
checks every round trip against mode 0 and pads only when necessary to prevent a
decodable stream from falling below the existing oracle diagnostic bound.

The complete 18-cell Experiment 019 regeneration exceeded the five-minute local
CPU smoke budget and was stopped without producing a result. A bounded codec
and provenance smoke used stabilized decoder seed 7, refiner seed 101, all 512
training clouds for that cell plus 512 FPS training constellations, and the 128
matched validation clouds for each method. This is not the complete factorial
and does not establish G-A2. Over its 256 validation rows, the validation-byte
selection chose `autoregressive`:

| Candidate | Mean fixed bytes | Mean mode-1 bytes | Mean mode-2 bytes | Fraction of fixed | Serialized shared model |
| --- | ---: | ---: | ---: | ---: | ---: |
| Octree | 50.000 | 51.996 | 52.344 | 1.047 | 10,434 B |
| Autoregressive | 50.000 | 51.996 | 52.070 | 1.041 | 13,091 B |

The selected candidate averaged 52.070 bytes for both the 128 FPS and 128
refiner rows (range 50--55 overall), so this bounded smoke expands rather than
compresses the fixed stream and fails the 40-byte G-A2 threshold. Its shared
integer arrays occupy 793,536 uncompressed bytes; the selected model hash is
`7bcdba756045c199421acdfe454c861a899e47beb30361a4065c8fc2cc1540ed`.
The 13,091-byte NPZ file SHA-256 is
`bd01f9575d1b0edb89e22c2827e371c7f5b7a5c0b9fb5399566b0ad8ffd28493`.
All 256 mode-2 streams were at or above the oracle diagnostic bound.

It used the same official-row and stability-config hashes listed above. Re-run
that bounded check with:

```bash
python scripts/resummarize_learned_entropy.py \
  --config configs/experiment_020_official_stability.json \
  --official-rows \
    artifacts/local/experiment_020_official_stability/official_per_cloud.jsonl \
  --device cpu \
  --inference-batch-size 32 \
  --max-refiner-cells 1 \
  --model-output /tmp/experiment_019_learned_entropy_model_smoke.npz \
  --output /tmp/experiment_019_validation_learned_entropy_smoke.json
```

Omit `--max-refiner-cells` for the predeclared complete factorial. The output
then records `complete_factorial: true`; limited runs explicitly record
`complete_factorial: false` and their cell count.

## Reproduction

The table was generated from `benchmark_metrics.json`, `per_cloud.jsonl`,
`gpcc_per_cloud.jsonl`, and every corresponding `stream.bin` under the existing
seed-7 artifact. The input hashes were:

- `benchmark_metrics.json`: `aa2cc3ae080c45c762f38a40101869e794eae32fd0744fd3448e567d65505378`
- `gpcc_per_cloud.jsonl`: `91c7ee753f39233cb617a237fe36a3a6f7aa7b6e19a02d7e7e67c29935ffe748`
- `per_cloud.jsonl`: `a96cdba072bed3d8292fe06aadebb21f485cd4f2314f5840b55923a0e30f448a`

Run the byte-exact re-summarizer without modifying the artifacts:

```bash
python scripts/resummarize_rate_accounting.py \
  --experiment-dir artifacts/local/experiment_017_modelnet40_multiseed/seed_7 \
  --amortize-parameter-sets-over 128 \
  --output /tmp/experiment_017_seed7_rate_accounting.json
```

The output contains all 13 G-PCC rate points for both recorded splits, not only
the validation-window rows shown above.
