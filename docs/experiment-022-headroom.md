# Experiment 022: multi-start Adam/STE inference headroom

Status: complete; Gate E passes (refiner on the encode-time/D1 Pareto front), but plain Adam-STE with 16 evaluations beats the refiner on D1 and D2.

## Question and hypothesis

Experiment 019 found that a 16-evaluation, FPS-initialized source-only Adam/STE
probe outperformed the recurrent refiner on 16 clouds per split. Experiment 022
tests whether that gap remains after increasing the per-cloud optimization
budget and adding independent starts.

The hypothesis is that the frozen recurrent refiner remains useful as
amortized inference: it should occupy a nondominated encode-time versus official
D1 point even if a slower per-cloud search reaches lower distortion. This is an
inference comparison at one fixed rate, not a general compression or
rate-distortion claim.

## Fixed protocol

- Reuse the six sealed Experiment 019 stabilized decoders and the three
  refiner checkpoints assigned to each decoder. Decoder parameters remain
  frozen during every search and refiner pass.
- Evaluate all 128 validation and 32 category-OOD clouds. The encoder-visible
  source sample is the only optimization target. Fresh resampling and source
  normals are evaluation-only data.
- Use `K=8`, 12-bit coordinates, and the existing canonical fixed-width
  bitstream. Every complete message is an unordered quantized coordinate set:
  a 14-byte header plus a 36-byte payload, or exactly 50 bytes.
- Start Adam/STE from deterministic FPS, deterministic k-means, and two
  independently seeded random strict subsets. K-means and all Adam results are
  free-coordinate constellations; FPS and the random initializations are not
  described as learned subsets after Adam moves their coordinates.
- Run independent Adam/STE searches at budgets 16, 64, and 256. One
  forward/backward source-Chamfer call counts as one decoder evaluation for
  every cloud in the batch. Serialization, post-search diagnostics, fresh
  Chamfer, and `pc_error` do not count against that budget.
- At each budget, select the multi-start candidate separately for each cloud
  and decoder using only its source-Chamfer search score. Fresh-resampling,
  D1, and D2 values are computed only after selection.
- Batch clouds for each decoder. Encode time includes initialization, search or
  refiner inference, exact quantization, and serialization. The full config
  records amortized per-cloud MPS and CPU columns; metric subprocess time is
  separate.

The serialized stream and distortion columns come from the declared quality
device. Timing on the other device is an independent deterministic execution
of the same arm whose candidate is discarded; the JSONL labels this explicitly.
This avoids presenting a cross-device timing rerun as the source of the scored
stream.

The Adam multi-start arm uses four times the declared per-start decoder budget.
For example, its 256-evaluation point costs 1,024 decoder evaluations per
cloud. It is an inference headroom bound, not an equally timed refiner arm.

## Outputs and resumability

`headroom_per_cloud.jsonl` contains one row for every decoder, cloud, start, and
budget, plus source-selected multi-start, FPS, and refiner rows. Each row stores
the canonical stream as hex, its SHA-256 and actual byte count, source and fresh
symmetric squared Chamfer, official D1/D2 results from `pc_error`, decoder
evaluation accounting, selected-start identity, and `encode_seconds_mps` and
`encode_seconds_cpu`.

The run manifest fixes the Experiment 019 config and metric hashes, data
membership, `pc_error` executable hash, model seeds, starts, budgets, timing
devices, and quality device. A resume is rejected if that manifest changes.
Completed JSONL rows are not appended again. Official metrics are also reused
when two arms serialize the same stream for the same decoder and cloud.

## Analysis

For source Chamfer, fresh Chamfer, D1, and D2, every individual Adam
start/budget and every multi-start budget is compared with FPS and with the
three-refiner factorial. The analysis reuses Experiment 020's paired
hierarchical category/cloud bootstrap and paired decoder draws. The common
refiner-seed factor is resampled for comparisons involving the refiner. The
reported effect is relative aggregate RMSE, even though JSONL rows store MSE.

The robust headroom quantity is reported per split and metric at the largest
multi-start budget:

```text
fraction_of_headroom_recovered =
    (FPS_RMSE - refiner_RMSE) / (FPS_RMSE - best_Adam_RMSE)
```

It is left undefined if the selected Adam arm does not improve on FPS. This
fraction is descriptive; the paired bootstrap comparisons carry uncertainty.

The Pareto table reports mean amortized encode seconds per cloud and aggregate
official D1 RMSE for FPS, the refiner factorial, every Adam start/budget, and
every multi-start budget. CPU and MPS fronts are computed separately.

## Predeclared gate and decision

The primary gate uses validation D1 and the quality device declared for the
run. It passes only if the aggregate refiner lies on the encode-time versus D1
Pareto front when considered jointly with FPS and all predeclared Adam starts
and budgets; equivalently, at least one budget must leave the refiner
nondominated. Validation selects the conclusion. Category OOD is transfer
evidence and cannot rescue a failed validation gate.

If the refiner is absent from that Pareto front, it is demoted to an ablation
in the paper and the headline method becomes Adam-selected coordinate
constellations. A passing gate supports describing the refiner as amortized
inference, but does not establish that its coordinates are a general-purpose
codec or that the fixed 50-byte rate is competitive elsewhere.

## Reproduction

Run the two-cloud, four-evaluation MPS smoke with:

```bash
.venv-train/bin/python -m pointconstellation.headroom_experiment \
  --config configs/experiment_022_headroom_smoke.json \
  --device mps
```

Run the full predeclared MPS-quality evaluation with both timing columns using:

```bash
.venv-train/bin/python -m pointconstellation.headroom_experiment \
  --config configs/experiment_022_headroom_modelnet40.json \
  --device mps
```

Both commands require the ignored Experiment 019 checkpoints, ModelNet40
manifest and meshes, and executable MPEG `pc_error` build. Generated streams,
metric scratch files, and results remain under `artifacts/local/` and are not
committed.

## Results

Complete six-decoder factorial: decoder seeds 7/17/29/41 on the MacBook (MPS) and 53/67 on homen-linux
(RTX 5070 Ti, CUDA), merged and aggregated once (18240 rows; timing column: CPU, the one device
common to both halves; MPS/CUDA encode times are retained per row). Paired hierarchical bootstrap, 10,000 draws.

**Gate E (primary gate) passes**: the aggregate refiner is non-dominated in encode seconds versus official D1
RMSE among FPS and every Adam start/budget arm — it is the only decoder-aware point below 30 ms per cloud.

### Headroom recovered by the refiner

| Split | Metric | FPS | Refiner | Best Adam (256) | Headroom recovered |
|---|---|---:|---:|---:|---:|
| validation | source_chamfer_mse | 0.1314 | 0.1092 | 0.0936 | 58.8% |
| validation | d1_mse | 316.7 | 264.4 | 235.6 | 64.4% |
| validation | d2_mse | 260.9 | 170.5 | 126.7 | 67.4% |
| ood | source_chamfer_mse | 0.148 | 0.1219 | 0.09928 | 53.5% |
| ood | d1_mse | 359.9 | 292.3 | 243.9 | 58.3% |
| ood | d2_mse | 315.9 | 209.2 | 136.2 | 59.4% |

### Refiner relative to each Adam arm (negative = refiner worse; 95% CI)

| Adam arm | Val D1 | Val D2 | OOD D1 | OOD D2 |
|---|---|---|---|---|
| `adam_ste:fps:budget_16` | -3.84% [-5.66, -2.34] | -25.13% [-33.75, -17.45] | -8.39% [-15.33, -3.54] | -32.29% [-47.31, -17.04] |
| `adam_ste:fps:budget_64` | -9.34% [-11.88, -7.39] | -30.20% [-39.08, -21.59] | -14.70% [-23.61, -8.16] | -40.59% [-57.94, -20.82] |
| `adam_ste:fps:budget_256` | -10.46% [-13.08, -8.45] | -31.28% [-40.80, -22.67] | -16.35% [-26.09, -9.13] | -42.51% [-59.77, -21.15] |
| `adam_multistart:budget_64` | -11.12% [-13.77, -9.07] | -33.74% [-44.30, -24.18] | -18.26% [-28.99, -10.54] | -51.21% [-84.92, -22.60] |
| `adam_multistart:budget_256` | -12.25% [-14.93, -10.07] | -34.56% [-45.31, -25.04] | -19.85% [-30.99, -11.48] | -53.53% [-87.61, -24.05] |

### Encode-time versus quality (validation, CPU timing)

| Arm (validation) | Encode s/cloud (CPU) | D1 RMSE | Pareto |
|---|---:|---:|---|
| `fps` | 0.001 | 316.7 | yes |
| `refiner` | 0.030 | 264.4 | yes |
| `adam_ste:random_seed_211:budget_16` | 0.280 | 266.9 |  |
| `adam_ste:random_seed_101:budget_16` | 0.287 | 268.6 |  |
| `adam_ste:fps:budget_16` | 0.301 | 254.6 | yes |
| `adam_ste:kmeans:budget_16` | 0.335 | 255.6 |  |
| `adam_ste:fps:budget_64` | 1.148 | 241.8 | yes |
| `adam_multistart:budget_16` | 1.203 | 249.1 |  |
| `adam_multistart:budget_64` | 4.623 | 237.9 | yes |
| `adam_multistart:budget_256` | 20.392 | 235.6 | yes |

### Reading

1. **The learned refiner is an amortization, not a quality mechanism.** With 16 decoder evaluations from the FPS
   start, plain Adam-STE already beats the refiner by 3.8% D1 [2.3, 5.7] and 25% D2 [17, 34] on validation, and
   by 8.4% / 32% on category OOD; 256-evaluation multi-start Adam beats it by 12% D1 and 35% D2. The refiner
   recovers 58--67% of the available headroom (D1/D2, validation) and 54--59% on OOD.
2. **It is on the Pareto front only as the cheap end**: ~30 ms per cloud on CPU versus ~300 ms for Adam-16 and
   ~1.15 s for Adam-64. Together with Experiment 021 (random best-of-16 through the decoder matches it on D1 and
   beats it on D2), the honest presentation is a time--quality curve in which the refiner is a 10x-cheaper
   approximation of decoder-aware search, and the paper's mechanism claim is *decoder-aware selection*.
3. **Multi-start adds little over a single FPS start** (Adam-256 single-start vs multi-start 256: ~2% D1), and
   random starts are slightly worse than FPS or k-means starts at equal budget; the search is not start-limited.
4. **D2 gains are much larger than D1 gains** for every Adam arm (e.g. Adam-64 vs FPS: +24% D1, +50% D2), which
   is consistent with the frozen decoder rewarding surface-aligned constellations that D1 under-weights.

Artifacts: `artifacts/local/experiment_022_headroom_merged/headroom_metrics.json` and `headroom_per_cloud.jsonl`
(merged); per-machine raw outputs under `experiment_022_headroom_modelnet40` (MacBook) and
`experiment_022_headroom_modelnet40_homen` (homen-linux). Comparison roles present: adam_vs_fps, refiner_vs_adam.

