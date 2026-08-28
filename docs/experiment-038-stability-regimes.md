# Experiment 038: stabilized decoder regimes

Status: complete; all eleven cells ran on EmpireAI with every contract check and gate passing.
this worktree.

## Question and protocol

Experiment 038 supplies the eleven stabilized-decoder dependencies missing from
the Experiment 033 `K x N` factorial. The full grid is `K in {4, 8, 16, 32}`
and `N in {1024, 2048, 4096}`. Experiment 019 already supplies `k8_n2048`;
Experiment 038 covers every other cell.

Each full cell retains the Experiment 019 ModelNet40 manifest, six decoder
seeds, three common refiner seeds, training cardinality curriculum
`[4, 8, 16, 32]`, epoch budgets, optimizer settings, and q=12 coordinate
lattice. Only `num_points`, `constellation_size`, and `output_dir` change. The
three N=2048 configs additionally declare the verified Experiment 019 decoder
artifact used for reuse. The manifest itself does not depend on sampled point
count, so its expected SHA-256 remains
`d44014dd2313b8815562cde9df2ba1927e1110fcfbc218428a7db39ef6b829ac`.

The complete per-cloud message remains one unordered, serialized, quantized
`K x 3` coordinate set. Decoder and refiner inputs do not receive features,
normals, labels, mesh identifiers, point order, or target-only samples.
Official rows use MPEG `pc_error` with `position_bits=12`, matching the
coordinate lattice. Actual stream bytes and bits per input point are recorded.

## Decoder reuse decision

The N=2048 decoder weights for K in `{4, 16, 32}` are legitimately reusable
from Experiment 019. `VariableConstellationDecoder` accepts any K up to its
configured maximum, represents K through a cardinality embedding, and was
trained by Experiment 019 with the exact repeating cardinality curriculum
`[4, 8, 16, 32]`. Changing evaluation K therefore does not change the decoder
architecture or claim that an unseen cardinality generalizes. The Experiment
038 runner verifies the source config, curriculum, dataset protocol, calibration
membership, checkpoint assignment, sealed selection, file SHA-256, and model
state hash before it skips decoder training. It then trains the K-specific
refiner cells and evaluates them into separate Experiment 038 artifact
directories. The result records the source artifact and hashes.

N=1024 and N=4096 decoders must be retrained. N determines the learned output
query tensor and the number of reconstructed points used during decoder
training. An N=2048 checkpoint cannot load into the N=4096 architecture, and
truncating its queries for N=1024 would be a post-hoc architecture projection,
not the declared Experiment 019 training protocol.

This reuse decision concerns shared decoder weights only. It does not make a
general compression claim, and the K-specific results remain separate cells.

## Mesh sampling and rate controls

`MeshSurfaceDataset` passes `num_points` directly to the area-weighted mesh
sampler for both source and fresh roles. Tests cover N=1024 and N=4096 and check
that source, training target, fresh sample, and normals have the declared
shape. Source and fresh samples retain independent role-derived deterministic
seeds. Changing N does not change manifest membership or train/calibration/
validation/category-OOD roles.

The generated configs retain `coordinate_bits=12`. The per-regime official
configuration is derived with `position_bits=12`; no `pc_error` grid setting is
inferred from N or K.

## Configuration and execution

Generate the eleven configs, or verify that the checked-in files are current:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  scripts/make_experiment_038_configs.py

/Users/hong/code/pointconstellation/.venv-train/bin/python \
  scripts/make_experiment_038_configs.py --check
```

Run one full cell end to end:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  scripts/run_experiment_038.py \
  --regime k4_n1024 \
  --device cuda
```

`--device` accepts exactly `cpu`, `mps`, or `cuda`. Distinct regimes write only
to their own stability and official-metric directories. A per-output advisory
lock rejects concurrent writers for the same regime; decoder reuse reads the
Experiment 019 artifact without modifying it. Complete stability results are
verified and resumed, while partial stability directories are rejected rather
than silently mixed with a new run. This permits separate regimes to run as
independent processes on one GPU, subject to available GPU memory.

For the checked smoke path:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  scripts/run_experiment_038.py \
  --regime k4_n1024 \
  --device cpu \
  --smoke
```

The smoke is explicitly non-inferential. It uses eight procedural clouds in
four disjoint roles, one decoder seed, one refiner seed, the full
`[4, 8, 16, 32]` training-cardinality curriculum, q=12 serialization, frozen
decoder checks, and official D1/D2 rows. It does not replace the six-by-three
full factorial.

Each cell writes `stability_metrics.json`, `per_cloud.jsonl`, checkpoints,
`official/official_per_cloud.jsonl`, `official/official_metrics.json`, and
`experiment_038_run.json`. The final file records stability time, official time,
and end-to-end wall-clock time. For scale reference, Experiment 019
`k8_n2048` took 1,408.75 seconds on Apple MPS as recorded in
`docs/experiment-019-stability.md`.

## Results

All eleven cells ran on EmpireAI H200s on 2026-08-27/28 (`--device cuda`, two
cells per GPU concurrently), each with the full six-decoder x three-refiner
protocol, 3,840 official rows (refiner and FPS-subset methods, validation and
category-OOD splits), 12-bit coordinates, and mode-0 streams. Every cell
reports `experiment_019_contract_passed`, `decoder_hashes_unchanged`,
`exact_stream_round_trip`, `source_only_selection`, and
`inferential_gate_eligible` true, and both the official-metric gate and the
selection-baseline gate pass in every cell. The hash-checked N=2048 reuse of
the Experiment 019 decoders was accepted for all three reuse cells.

| Cell | Decoder action | Stability wall s | Official wall s | Bytes | Refiner val D1 / D2 (dB) | Refiner OOD D1 / D2 (dB) | FPS val D1 / D2 (dB) | Contract | Gates |
|---|---|---:|---:|---:|---|---|---|---|---|
| `k4_n1024` | train | 453 | 502 | 32 | 27.31 / 32.89 | 26.78 / 32.06 | 26.45 / 30.66 | pass | pass |
| `k8_n1024` | train | 442 | 489 | 50 | 27.76 / 33.48 | 27.24 / 32.46 | 26.95 / 31.13 | pass | pass |
| `k16_n1024` | train | 441 | 487 | 86 | 28.33 / 33.92 | 28.07 / 33.14 | 27.24 / 31.59 | pass | pass |
| `k32_n1024` | train | 426 | 477 | 158 | 29.04 / 34.61 | 28.65 / 33.57 | 27.77 / 32.20 | pass | pass |
| `k4_n2048` | reuse Experiment 019 | 430 | 560 | 32 | 28.45 / 33.31 | 27.77 / 31.98 | 27.35 / 30.60 | pass | pass |
| `k16_n2048` | reuse Experiment 019 | 429 | 564 | 86 | 29.29 / 34.30 | 28.96 / 33.29 | 28.08 / 31.59 | pass | pass |
| `k32_n2048` | reuse Experiment 019 | 397 | 595 | 158 | 29.69 / 35.01 | 29.26 / 34.12 | 28.22 / 31.93 | pass | pass |
| `k4_n4096` | train | 1861 | 1590 | 32 | 29.06 / 34.33 | 28.56 / 33.18 | 27.91 / 31.32 | pass | pass |
| `k8_n4096` | train | 1862 | 1590 | 50 | 29.47 / 34.76 | 28.92 / 33.65 | 28.34 / 31.75 | pass | pass |
| `k16_n4096` | train | 1855 | 1660 | 86 | 29.77 / 35.10 | 29.34 / 33.97 | 28.55 / 32.24 | pass | pass |
| `k32_n4096` | train | 1856 | 1675 | 158 | 29.93 / 35.84 | 29.50 / 34.87 | 28.56 / 32.72 | pass | pass |

D1/D2 values are means of the per-cloud official PSNR rows (`d1_psnr_db`,
`d2_psnr_db`); paired bootstrap comparisons are in each cell's
`official/official_metrics.json`. The existing Experiment 019 `k8_n2048`
result remains its own twelfth grid cell and is not duplicated here.

### Reading

1. **K moves fidelity very little at fixed N.** At N=2048 the refiner's
   validation D1 rises only from 28.45 dB (K=4, 32 B) to 29.69 dB (K=32,
   158 B): five times the bytes for 1.2 dB. The same holds at N=1024 (27.31 to
   29.04 dB) and N=4096 (29.06 to 29.93 dB). This is the cross-regime form of
   the Experiment 025 finding that quality saturates with K.
2. **N moves fidelity more than K does.** At K=4, going from N=1024 to 4096
   adds 1.75 dB D1 at identical bytes; denser sampling of the same surface makes
   the decoder-aware search easier, which is consistent with the constellation
   being a surface code rather than a point-sample code.
3. **The decoder-aware refiner beats the FPS subset in every cell**, by
   0.9-1.5 dB D1 and 2.2-3.3 dB D2, and the selection-baseline gate passes in
   all eleven cells; the gap is widest for D2 and at large N.
4. Wall time: N=1024 and reuse cells take about 15-17 min end to end; N=4096
   cells about 58 min, dominated equally by decoder training and official
   metrics.

Artifacts: `artifacts/local/experiment_038_stability_<regime>/` and the
summary `artifacts/local/experiment_038_summary.json` on the cluster
(`~/LLM_research/pointconstellation-tracks-b`). Experiment 033 can now run its
full factorial against these cells.
