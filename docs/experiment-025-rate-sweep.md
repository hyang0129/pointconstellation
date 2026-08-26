# Experiment 025: stabilized Adam-STE rate sweep

Status: implemented; the full ModelNet40 curve is pending.

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
