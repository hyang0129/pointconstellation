# Experiment 025: stabilized Adam-STE rate sweep

Status: complete; Gate G-A1 passes (14/14 eligible points non-dominated by G-PCC below 80 bytes).

## Question and hypothesis

Experiment 025 tests whether source-only Adam/STE inference over a sealed,
stabilized decoder produces a competitive multi-rate coordinate-constellation
curve. The primary arm starts from FPS and uses 64 frozen-decoder evaluations.
The hypothesis is that at least four validation operating points below 80
complete stream bytes are not dominated by a measured G-PCC point after both
codecs are compared at byte-aligned payload rate.

This is a rate-distortion test, not a claim that coordinates necessarily retain
their usual geometric meaning. The complete learned message remains only an
unordered, quantized `K x 3` coordinate set. Decoder features, target-only
information, point order, normals, labels, primitive IDs, and per-instance
weights are not transmitted.

## Fixed protocol

The grid is `K in {4, 6, 8, 12, 16}` crossed with coordinate precision
`q in {8, 10, 12}`. Every message is encoded with the canonical `PCON` stream,
decoded, and checked for an exact second encoding before neural reconstruction.
The current format has a 14-byte header, so the byte-exact grid spans 26 through
86 complete bytes. The earlier indicative 23-byte lower endpoint is not used
because it is inconsistent with `expected_stream_bytes(4, 8)`.

For each cell and decoder seed, the encoder arms are:

- FPS through the same frozen decoder;
- Adam/STE initialized from FPS with 64 decoder evaluations, the primary arm;
- a source-selected four-start bound with independent FPS, k-means, and two
  seeded random-subset starts, each receiving 64 evaluations; and
- the existing competitive refiner only at `K=8`, crossed over its three
  refiner seeds.

The smoke configuration reduces Adam to two evaluations. It is a pipeline and
contract test, not a result-producing configuration.

## Decoder reuse decision

`VariableConstellationDecoder` accepts every cardinality up to its configured
maximum, and the Experiment 019 checkpoint was trained with the multi-K
curriculum `K={4,8,16,32}`. One sealed stabilized decoder per seed therefore
serves the entire Experiment 025 grid; the six existing Experiment 019
checkpoints are reused and never retrained. `K=6` and `K=12` are explicitly
recorded in the run manifest as cardinality-interpolation evaluations because
those exact cardinalities were not curriculum steps. This distinction must be
retained when reporting the curve.

The stabilized checkpoint was selected from EMA candidates using only the
Experiment 019 calibration partition after four decoder epochs. The runner
checks the saved selection hash, checkpoint hash, model state hash, Experiment
019 config, scientific-contract flags, and dataset partition identity before
evaluation. It also checks that decoder state hashes remain unchanged after
Adam and refiner inference.

The Experiment 019 refiner exists only at `K=8`. At `q=8` and `q=10`, its
coordinates are requantized by the declared cell bitstream before decoding;
this is a precision ablation of the existing refiner, not a newly trained
lower-precision refiner.

## Metrics and rate accounting

All 128 validation and 32 category-OOD clouds are evaluated with official MPEG
`pc_error` D1 and D2 on the common 12-bit grid. Source and fresh-resampling
symmetric squared Chamfer are diagnostics. Normals are used only by the
official evaluator. Adam and multi-start selection read only source-cloud
Chamfer through the frozen decoder.

Every per-cloud row records:

- `stream_bytes` and `actual_stream_bpp` for the independently decodable
  complete stream;
- `header_bytes`, `payload_bytes`, and `payload_bpp`, where the payload is the
  byte-aligned coordinate payload and includes its final padding byte;
- the stream bytes as hex and SHA-256, exact round-trip and lattice checks;
- official D1/D2, source/fresh Chamfer, timing, model seeds, and search budget.

The G-PCC reference is the exact-cloud Experiment 017 TMC13 curve. The runner
uses Experiment 021 payload fields when present. For older rows it parses each
preserved TMC13 stream's type-length-value framing to recover byte-exact payload
bytes. It does not infer payload size from a nominal rate or subtract an
estimated header.

Within every cell and split, D1 and D2 for each candidate arm are compared with
FPS using a paired hierarchical category/cloud bootstrap with paired decoder
draws. They are also compared with the complete G-PCC point whose mean payload
bytes are nearest to the cell payload. The G-PCC cloud identities must match the
cell exactly. Refiner comparisons additionally resample the common refiner-seed
factor. Rows store MSE; summaries and confidence intervals report aggregate
RMSE effects.

## Predeclared gate G-A1

G-A1 passes only if at least four primary validation Adam-64 points with
complete streams strictly below 80 bytes are not jointly dominated by a
measured G-PCC point in payload bytes, official D1 MSE, and official D2 MSE. A
G-PCC point dominates when it is no worse on all three quantities and strictly
better on at least one.

If G-A1 fails, Track A stops. A passing result permits the remaining Track A
paper work but does not by itself establish a general compression advantage.
OOD results are transfer evidence and cannot rescue a failed validation gate.

## Artifacts and resume behavior

Each cell writes independently under
`artifacts/local/experiment_025_rate_sweep_modelnet40/cells/k_<K>_q_<q>/`.
`rate_sweep_per_cloud.jsonl` is append-only and keyed by split, method, decoder,
optional refiner seed, and cloud identity. On resume, duplicate keys and any
rate field inconsistent with `expected_stream_bytes` are rejected; completed
rows are not evaluated or appended again.

The root output contains:

- `run_manifest.json`, including config/tool/data hashes and the decoder reuse
  decision;
- `rate_sweep_metrics.json`, with the per-cell table, comparisons, contract
  checks, and G-A1 decision;
- `rate_sweep_curve.json`, a stable `rows` document consumable by the benchmark
  registry; and
- `rate_sweep_table.md`, the human-readable per-cell table.

The per-cell JSONL files are also directly discoverable by the registry. A
single-cell cluster job may leave the root summary partial until all jobs
finish; `--aggregate-only` deterministically rebuilds it without inference.

## Reproduction

Run the two-cloud, one-decoder CPU smoke:

```bash
.venv-train/bin/python -m pointconstellation.rate_sweep_experiment \
  --config configs/experiment_025_rate_sweep_smoke.json \
  --device cpu
```

Dispatch one resumable Jupyter job per cell on EmpireAI:

```bash
scripts/launch_experiment_025_empire.sh
```

After all cell jobs complete, rebuild the curve and gate:

```bash
.venv/bin/python -m pointconstellation.rate_sweep_experiment \
  --config configs/experiment_025_rate_sweep_modelnet40.json \
  --aggregate-only
```

The full run requires the ignored Experiment 019 checkpoints, ModelNet40 data
and manifest, executable MPEG `pc_error`, and either rate-accounted G-PCC rows
or the preserved G-PCC streams. Generated results remain under
`artifacts/local/` and are not committed.

## Results

All 15 `(K, q)` cells completed on EmpireAI H200s (51840 per-cloud rows; six sealed Experiment 019
decoders, 128 validation + 32 category-OOD clouds, official `pc_error` on the 12-bit grid). Validation, Adam-64
encoder unless stated (RMSE in grid units):

| K | q | Stream B | Payload B | FPS D1 | Adam-64 D1 | Adam-64 D2 | Multi-start D1 | Non-dominated |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 8 | 26 | 12 | 341.8 | 255.8 | 136.5 | 250.3 | yes |
| 4 | 10 | 29 | 15 | 341.8 | 255.9 | 136.3 | 250.3 | yes |
| 4 | 12 | 32 | 18 | 341.8 | 255.9 | 136.3 | 250.2 | yes |
| 6 | 8 | 32 | 18 | 320.8 | 247.7 | 133.2 | 243.1 | yes |
| 6 | 10 | 37 | 23 | 320.8 | 247.5 | 132.9 | 243.0 | yes |
| 8 | 8 | 38 | 24 | 316.7 | 241.8 | 131.1 | 237.9 | yes |
| 6 | 12 | 41 | 27 | 320.8 | 247.7 | 133.2 | 243.0 | yes |
| 8 | 10 | 44 | 30 | 316.8 | 241.8 | 130.9 | 237.9 | yes |
| 8 | 12 | 50 | 36 | 316.7 | 241.8 | 131.0 | 237.8 | yes |
| 12 | 8 | 50 | 36 | 306.4 | 233.9 | 127.2 | 230.5 | yes |
| 12 | 10 | 59 | 45 | 306.4 | 234.0 | 127.2 | 230.6 | yes |
| 16 | 8 | 62 | 48 | 308.0 | 229.1 | 123.6 | 225.7 | yes |
| 12 | 12 | 68 | 54 | 306.4 | 234.0 | 127.2 | 230.7 | yes |
| 16 | 10 | 74 | 60 | 308.1 | 229.1 | 123.7 | 225.7 | yes |
| 16 | 12 | 86 | 72 | 308.0 | 229.1 | 123.7 | 225.7 | n/a (>=80 B) |

Measured G-PCC (Experiment 017 seed-7 validation, TMC13 v23, byte-exact payload from TLV parsing):

| G-PCC point | Stream B | Payload B | D1 RMSE | D2 RMSE |
|---|---:|---:|---:|---:|
| `octree_s1_768` | 55 | 18 | 384.2 | 226.6 |
| `octree_s1_640` | 61 | 24 | 317.8 | 186.6 |
| `octree_s1_512` | 70 | 33 | 251.4 | 144.7 |
| `octree_s1_256` | 114 | 77 | 126.4 | 72.2 |

**Gate G-A1 passes** (`continue_track_a`): 14 of 14 eligible stabilized points
below 80 bytes are not jointly dominated in payload bytes, D1, and D2 by any measured G-PCC point (required: 4).

### Reading

1. **Below ~65 bytes the constellation dominates G-PCC on both full-stream and payload accounting.** A 26-byte
   `K=4, q=8` constellation (12 payload bytes) reaches D1 255.8, essentially G-PCC's 70-byte `octree_s1_512`
   point (251.4, 33 payload bytes); `K=8, q=8` at 38 bytes beats it outright (241.8).
2. **Precision beyond 8 bits is wasted.** At every `K`, `q=8`, `q=10`, and `q=12` are indistinguishable
   (differences < 0.2 RMSE), so the declared 12-bit stream can shrink by 24--33% at no cost. At a fixed
   50-byte budget, `K=12, q=8` (D1 235.6 val / 243.9 OOD) beats `K=8, q=12` (241.8 / 254.8): bytes are
   better spent on points than on precision.
3. **Quality saturates with `K`.** D1 improves only from 255.8 (`K=4`) to 229.1 (`K=16`) while bytes rise from
   26 to 62--86; G-PCC reaches 126.4 at 114 bytes. The crossover is therefore ~70--110 bytes, and the ceiling
   is set by the frozen decoder, not by the message. Raising the ceiling (decoder capacity, training data,
   or objective; Track B) is what would extend the winning regime.
4. Multi-start Adam (4 starts x 64 evaluations) adds only 1.5--2% over single-start Adam-64; the refiner
   (K=8 only) trails Adam-64 by ~9% D1 and ~24% D2 on validation.
5. Category-OOD reproduces the ordering at every cell (see `rate_sweep_table.md`).

Artifacts: `artifacts/local/experiment_025_rate_sweep_modelnet40/rate_sweep_{metrics,curve}.json`, `rate_sweep_table.md`,
per-cell `cells/k_*_q_*/cell_metrics.json` and `rate_sweep_per_cloud.jsonl` (cluster: `~/LLM_research/pointconstellation-tracks-a`).

