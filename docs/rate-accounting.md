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
