# Experiment 041 post-decode acceleration

## Outcome

Experiment 041 now runs label transfer and the k-NN normal-manifold search with
chunked PyTorch `cdist`/`topk` kernels on the requested benchmark device, scores
decoded clouds in batches, and evaluates rank metrics and hierarchical
bootstraps on vectorized dense arrays. The scientific protocol, scorer
parameters, serialized rates, `ScoredCloud` schema, row ordering, and resume
identity remain unchanged.

The requested real CPU smoke fell from 61.024312 seconds before the change to
11.906599 seconds after the final numerically stable implementation, a 5.13x
end-to-end speedup even though codec encode work is unchanged. Its 900
`defect_per_cloud.jsonl` rows contain 13,548 numeric values. The maximum
absolute pre/post difference was `1.1102230246251565e-16`; all nonnumeric fields
were identical. The summaries, G-C2 gate, and contract checks also matched the
pre-change result to `1e-9`.

## Full-parameter synthetic profile

The profile used the full configuration values without reading or writing
`data/` or `artifacts/`:

- 128 independently generated normal references, each initially 2,048 points;
- the real seeded fit subsampling of 512 points per reference;
- 2,048-point decoded/query clouds;
- three scorer seeds, three candidate references, `k=3`, and distance chunks
  of 512 points;
- tied score fixtures for AUROC/AUPRC;
- 160 cloud identities, six control/defect conditions, and 10,000 bootstrap
  draws for report-scale projections.

The pre-change profile on the available Apple CPU host measured:

| Stage | Measured | Full-run extrapolation |
|---|---:|---:|
| Fit all three scorers | 0.030751 s | 0.030751 s |
| Scorer inference | 0.019946 s/scored row | 2,154.13 s at 108k rows |
| Nearest-target label transfer | 0.221481 s/decode | 7,973.32 s at 36k decodes |
| Point AUROC + AUPRC | 0.002192 s/scored row | 236.69 s at 108k rows |
| Cloud AUROC + AUPRC on 960 rows | 0.001023 s/pair | report-dependent |
| Nine stratification filters over 108k rows | 0.059270 s | negligible |
| One cloud-AUROC bootstrap, 100 draws/960 rows | 1.099215 s | 109.92 s at 10k draws |

The nested legacy bootstrap rescanned Python row objects for every sampled
seed, cloud, draw, summary, and metric. A single 10,000-draw metric therefore
projected to about 110 seconds, while the complete report requests hundreds of
such metrics. This is consistent with the observed 21-hour CPU tail. Label
transfer was the other hidden dominant cost: it performed a Python `lexsort`
for every decoded query point and was charged to codec decode timing.

The final optimized full-parameter profile measured:

| Stage | Measured | Conservative extrapolation |
|---|---:|---:|
| Fit and device-materialize all three scorers | 0.042977 s | 0.042977 s |
| Batched scorer inference | 0.011329 s/scored row | 1,223.59 s at 108k rows |
| Device label transfer | 0.007194 s/decode | 258.98 s at 36k decodes |
| Vectorized point AUROC + AUPRC | 0.000401 s/scored row | 43.33 s at 108k rows |
| Vectorized cloud AUROC + AUPRC on 960 rows | 0.000219 s/pair | report-dependent |
| Nine stratification filters over 108k rows | 0.063169 s | negligible |
| One vectorized cloud-AUROC bootstrap, 10k draws/960 rows | 1.140354 s | measured directly |
| Complete one-arm/rate, two-split, 18-summary report | 29.470280 s | 736.76 s for 25 arm/rate cells |

The resulting conservative post-decode projection is about 2,272 seconds, or
37.9 minutes, for the requested 108k scored-row workload (108k scoring/metric
rows, 36k label transfers, the complete 25-cell report, gate, and
stratification). The literal topology in the supplied config is 160 clouds x 6
conditions x (24 coded arm/rate cells + raw) x 3 scorer seeds = 72,000 scored
rows; that topology projects to about 29.4 minutes on this CPU profile. Both are
well under one hour before applying CUDA acceleration. This host has no CUDA
device, so no RTX timing is claimed; an RTX run will use the configured CUDA
device for the distance/top-k work.

## Implementation

- Each scorer seed samples exactly the same reference clouds and points as
  before, then materializes those references on the benchmark device once.
- Descriptor-based candidate selection remains stable float64 NumPy code.
- One chunked `torch.cdist` matrix now supplies forward and reverse k-NN
  neighbor selection. Queries/candidates are batched by cardinality and
  `torch.topk` maintains reverse neighbors across chunks.
- Only the selected neighbors are re-evaluated with the legacy float64
  arithmetic. This inexpensive refinement preserves exact CPU point scores on
  random fixtures while retaining device acceleration for the quadratic
  search.
- Label transfer canonicalizes references once, then uses chunked device
  `cdist` and `argmin`; its canonical tie rule is unchanged.
- AUROC uses vectorized average ranks for exact ties. AUPRC uses vectorized
  grouped thresholds and cumulative positive counts. Degenerate-label behavior
  remains unchanged.
- Bootstrap rows are packed once into seed x cloud x condition arrays. The
  legacy alternating seed/cloud random draw sequence is retained, but each
  chunk of draws is evaluated together rather than rescanning Python objects.
- Decoded units are accumulated into batches of 8 on CPU or 16 on an
  accelerator. Rows are still appended in the canonical unit-then-scorer order.
- The `scoring` progress stage reports scored rows, total rows, the active
  unit, and the persisted `defect_per_cloud.jsonl` path. Each completed batch
  is flushed and synced after writing its individual scored-row lines. A
  restart loads those rows, validates the canonical prefix, and skips their
  scorer computation; a crash can lose at most the active 8/16-unit batch.
- Timing now separates `label_transfer_seconds`,
  `scorer_inference_seconds`, and `point_metrics_seconds`; the existing
  `official_metrics_seconds` remains the aggregate scorer/metric time.

The old NumPy distance, rank-loop, and label-transfer algorithms are retained
only as test references where needed. The benchmark path no longer calls them.

## Equivalence and resume coverage

`test_optimized_scorer_transfer_and_metrics_match_reference_to_float64`
compares the optimized scorer, canonical label transfer, AUROC, and AUPRC with
the old algorithms on random inputs, duplicate/equidistant points, exact score
ties, all-negative labels, and all-positive labels. Scorer outputs are checked
at absolute tolerance `1e-9`; the sampled CPU fixtures are bit-identical.

The incremental-resume test now also verifies that scoring progress exposes the
persisted row path, begins with the exact resumed scored-row count, reaches its
total, and reconstructs the byte-identical canonical JSONL after a partial
restart. A complete real-smoke resume reported 900/900 persisted scoring rows,
performed no decode or scoring work, finished in 1.049 seconds, and retained
the identical JSONL SHA-256
`6c6c96817950bf4482b0b6bd9a288a0b9b43c697cf3c450ab5a8abb0a8853ccb`.

## Verification

Required lint:

```text
/Users/hong/code/pointconstellation/.venv-train/bin/python -m ruff check src tests
All checks passed!
```

Required full test suite:

```text
PYTHONPATH=src /Users/hong/code/pointconstellation/.venv-train/bin/python -m pytest -q
353 passed, 6 skipped
```

The six skips are the existing optional-environment cases for Matplotlib,
Draco, and `h5py`.

Required real smoke:

```text
PYTHONPATH=src /Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.defect_anomaly_benchmark \
  --config configs/experiment_041_defect_anomaly_smoke.json \
  --device cpu \
  --output-dir /tmp/exp041-fast-smoke
```

The clean final run completed all 300 evaluation units and 900 scorer rows in
11.906599 seconds with every contract check true. Its rows, summaries, and gate
were compared programmatically with the pre-change clean smoke at
`/tmp/exp041-fast-smoke-prechange` and matched to `1e-9` as described above.

No contents of `data/` or `artifacts/` were modified. No commit or push was
made.
