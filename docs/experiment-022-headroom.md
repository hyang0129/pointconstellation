# Experiment 022: multi-start Adam/STE inference headroom

Status: implemented; the full ModelNet40 run is pending.

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
