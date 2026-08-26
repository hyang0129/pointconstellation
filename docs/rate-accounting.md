# End-to-end rate accounting

This note re-summarizes the existing Experiment 017 seed-7 artifacts. It is an
accounting correction, not a new training or codec run. The validation split
has 128 ModelNet40 clouds of 2,048 source points each.

## Definitions

`actual_stream_bpp` remains the complete serialized file size in bits divided
by the source point count. `payload_bpp` uses the byte-aligned payload size, so
padding in the final payload byte is still counted.

Mesh experiments normalize an object with its bounding-box center and maximum
vertex radius. That transform is not shared. New `PCON` streams therefore use
the existing domain byte to signal an 8-byte normalization payload: three
binary16 center values followed by one positive binary16 isotropic scale.
Feature-codec, G-PCC, and published external learned-codec evaluations store
the same 8 bytes as per-object side information. `header_bytes`, `payload_bytes`,
and `normalization_bytes` sum to the full per-object byte count. Old domain-1
`PCON` streams remain decodable and have `normalization_bytes = 0`; they
represent only the shared pre-normalized domain.

Binary16 rounds each normal center component and the scale with relative error
at most `2^-11`; subnormal values have an absolute spacing of `2^-24`. Values
outside the finite binary16 range and scales that round to zero are rejected.
Original-frame reconstruction uses the serialized center and scale, so this
rounding is included in every `original_frame_*` distortion field (named
`original_frame_official_*` in the standardized benchmark). The unprefixed
official fields continue to measure the shared normalized frame.
For `pc_error`, original-frame source and reconstruction coordinates are mapped
to its integer grid with the encoder's declared mesh box; only the decoded
reconstruction uses the rounded transmitted transform.

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

## Shared-decoder amortization

For a complete per-object stream of `s` bytes, a decoder state dict of `m`
bytes, `N=2048` source points, and a corpus of `n` independently coded objects,
the reported total rate is

```text
amortized_bpp(n) = 8 * (s + m / n) / N.
```

The state-dict sizes below are exact `torch.save` file sizes regenerated from
the Experiment 019 seed-7 stabilized decoder and Experiment 018 seed-7 feature
decoder. Floating tensors are stored entirely as the named precision; integer
buffers retain their native type. Encoder and refiner weights are excluded
because they are not needed for decoding. The per-object point is `K=8,q=12`.
This is why the fp32 deployment number is smaller than the earlier approximate
405 KB inventory, which combined decoder and encoder-side refiner checkpoints.
The G-PCC row uses the measured Experiment 017 validation
`octree_s1_640` mean. The `pcc_geo_cnn_v2` row uses its smallest released-rate
smoke stream and, conservatively, the complete five-point release bundle because
the retained local Experiment 020 artifact does not contain the runner's
single-checkpoint `model_bytes` row. It is an upper-bound accounting row, not a
single-rate deployment size.

| Codec / model representation | Object bytes without normalization | Object bytes with normalization | Model bytes | bpp at n=128 | n=672 | n=2,468 | n=10k | n=100k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constellation decoder, fp32 | 50.000 | 58.000 | 337,942 | 10.539734 | 2.190976 | 0.761443 | 0.358571 | 0.239763 |
| Constellation decoder, fp16 | 50.000 | 58.000 | 174,294 | 5.545593 | 1.239711 | 0.502428 | 0.294646 | 0.233371 |
| Feature-codec decoder, fp32 | 50.000 | 58.000 | 871,110 | 26.810730 | 5.290213 | 1.605320 | 0.566840 | 0.260590 |
| Feature-codec decoder, fp16 | 50.000 | 58.000 | 438,470 | 13.607605 | 2.775332 | 0.920555 | 0.397840 | 0.243690 |
| `pcc_geo_cnn_v2`, native release bundle upper bound | 1,751.000 | 1,759.000 | 5,523,313,425 | 168,565.020050 | 32,113.185181 | 8,748.946891 | 2,164.415400 | 222.625524 |
| G-PCC full independently decodable stream | 61.281 | 69.281 | 0 | 0.270630 | 0.270630 | 0.270630 | 0.270630 | 0.270630 |
| G-PCC sequence accounting, SPS/GPS shared | 61.281 | 69.281 | 0 | 0.146606 | 0.145816 | 0.145681 | 0.145642 | 0.145631 |

The last row shares the measured mean 32 SPS/GPS bytes and retains the 8-byte
normalization for every object. It is not an independently decodable stream.
New G-PCC summaries report both the zero-model `amortized_bpp` table (identical
to the full-stream rate) and `sequence_amortized_bpp`.

Against G-PCC's 0.270630-bpp full per-object `octree_s1_640` stream, the
normalization-complete constellation falls below the G-PCC rate after 29,957
objects with the fp32 state dict and after 15,450 objects with fp16. Thus it is
still above G-PCC at `n=10k` but below it at the next requested corpus size,
`n=100k`. This crossover is rate accounting for these measured streams; it is
not a rate-distortion superiority claim.

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
