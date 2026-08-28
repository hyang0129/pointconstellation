# Experiment 040: heuristic selective pass-through

Status: implementation complete; full results not yet run.

## Question and predeclared hypothesis

Experiment 040 asks whether allocating part of a fixed coordinate-payload budget
to source points with high local irregularity makes source-to-reconstruction D1
error less dependent on irregularity than allocating all payload coordinates to
a uniformly larger free-coordinate constellation. The primary hypothesis is
that source-only curvature selection with 50% of coordinate slots preserved
reduces the slope of D1 RMSE across source-curvature quintiles without a material
aggregate D1 penalty.

This is an explicitly labeled selective-coordinate ablation. Its complete
per-cloud message is two unordered coordinate sets rather than the strict single
unordered `K x 3` learned message. It must not be reported as a geometry-only
constellation result.

## Message and rate accounting

The fixed selective stream stores `(K1 constellation points, K2 preserved
points)` on the common `[-1, 1]`, 12-bit lattice. Each set is independently
lexicographically sorted. The header stores `K1` and `K2`; the first `K1` points
are decoder inputs and the last `K2` points are passed through. There is no
per-point type channel. Decoding is

```text
X_hat = G(Z_constellation) union Z_preserved.
```

The decoder appends the decoded preserved coordinates without numeric
conversion and asserts zero preservation error. A zero-length preserved group
uses the existing fixed or Rice stream byte-for-byte. A zero-length
constellation is a pure pass-through message.

For fixed coding, payload bytes are exactly
`ceil(3 (K1 + K2) q / 8)`. A selective header is 16 bytes, two bytes larger than
the ordinary constellation header because it carries both counts. Results
separately report the coordinate-payload budget, actual fixed payload bytes,
complete fixed stream bytes, and the byte count of an actual decodable selective
Rice stream. The Rice variant includes its four-byte first-block length; it is
not an entropy estimate.

The shared frozen decoder checkpoint size and parameter count are recorded
separately and are not included in per-cloud rate. Any paper claim must add an
explicit corpus-size amortization rather than silently treating that model as
free.

The primary 12-bit payload budgets are 40, 52, 64, 78, 96, and 110 bytes. They
give uniform cardinalities 8, 11, 14, 17, 21, and 24. Selective allocations use
the same total cardinality and round 25%, 50%, or 75% of it to preserved points.
The diagnostic 8-bit row is `K=4`: 12 payload bytes plus the ordinary 14-byte
header gives the declared 26-byte complete stream. It is a uniform-only
precision diagnostic and is excluded from G-C1.

G-PCC uses the existing Experiment 017 harness and its checked TMC13 rate-point
arguments, but re-encodes the current Experiment 040 source sample. Older
reconstructions are not reused because they may come from a different finite
resampling of the same mesh. For each cloud, the runner measures the complete
fresh G-PCC frontier, chooses the rate with the nearest actual geometry-brick
payload, recomputes official metrics on the current source, and keeps the
complete stream and component byte counts. It does not interpolate codec rates.
The byte delta remains visible in the per-cloud row, so a coarse G-PCC grid
cannot be described as an exact match.

## Source-only selection scores

All four heuristic scores read only the finite source cloud. They are
permutation-equivariant and deterministic.

- `curvature` is local PCA surface variation, the smallest covariance
  eigenvalue divided by the trace over 16 nearest neighbors.
- `density` is the absolute log deviation of the local mean neighbor radius
  from the cloud median.
- `decoder_residual` is the one-way source distance to the Chamfer-optimized
  constellation-only decode for the same `K1`.
- `boundary` combines one-sided neighborhood-centroid displacement and
  unoriented local-normal disagreement.

The last score is a finite-sample, source-only boundary/sharp-feature proxy. It
is not the analytic mesh-boundary mask used by the procedural part of Experiment
031. Scores select the top `K2` points greedily with 0.02 minimum source-space
spacing. A coordinate-keyed deterministic random score supplies the matched
random-`K2` control. Every `K1` constellation is independently optimized with
the Experiment 022/025 FPS-start Adam-STE search against source Chamfer. The
stabilized Experiment 019 decoder is loaded from its sealed checkpoint, frozen,
hash-checked before and after evaluation, and never retrained.

## Metrics and uncertainty

Every learned/selective arm is serialized and decoded before evaluation.
Per-cloud rows contain official symmetric D1 and D2 from `pc_error`, source D1
by precomputed source-curvature quintile, source-to-reconstruction p95 and p99
Euclidean error, boundary-proxy recall, PCA-thinness-proxy recall, normal
consistency, and encode/decode time. Quintile membership is computed once per
source cloud and reused by every arm. All five bins cover the complete source
cloud.

Normal consistency estimates unoriented reconstruction normals by local PCA and
matches them to source normals. Boundary and thin-structure recalls use the top
quintile of their declared source-only scores and the fixed 0.02 tolerance.
These ModelNet40 diagnostics are finite-sample proxies, not analytic continuous
surface claims. The smoke skips normal consistency to remain bounded; the full
configuration enables it.

The full run crosses six frozen decoder seeds with validation and category-OOD
clouds. G-C1 uses a paired hierarchical bootstrap: decoder seeds are resampled
as a paired factor and categories then clouds are resampled while keeping
uniform and selective measurements paired. The checked full configuration uses
10,000 draws and 95% percentile intervals. A one-seed smoke writes G-C1 as
pending rather than deciding it.

For every selective and random-control arm, the results JSON also contains
paired intervals against uniform `K` for official D1 RMSE, official D2 RMSE,
and mean per-cloud p99 error. These comparisons use the same paired
seed/category/cloud resampling structure; they are diagnostics and do not alter
the predeclared gate.

## Gate G-C1

The primary arm is curvature selection with a 50% preserved allocation. At each
12-bit payload point, fit a line to the five pooled quintile D1 RMSE values. A
positive slope reduction means the selective arm is flatter than uniform `K`.

G-C1 passes only if:

1. at least four of the six validation payload points have a strictly positive
   lower 95% paired-bootstrap bound for slope reduction;
2. at those same passing cells, the upper 95% bound on aggregate official D1
   RMSE degradation is at most 5%; and
3. the point estimate of slope reduction is positive at at least four of six
   category-OOD payload points.

The runner writes the definition, every rate-level interval, counts, decision,
and `passes` value under `gate_g_c1` in `selective_metrics.json`. The gate is not
evaluable with fewer than three decoder seeds, without both splits, or without
the complete six-point 12-bit grid.

## Commands

Run the bounded 8-cloud, one-decoder CPU smoke:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.selective_experiment \
  --config configs/experiment_040_selective_smoke.json \
  --device cpu
```

Run the predeclared full experiment after the smoke passes:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.selective_experiment \
  --config configs/experiment_040_selective.json \
  --device mps
```

`--device cuda` is also accepted. Both commands require the ignored Experiment
019 checkpoints, ModelNet40 manifest/data, official `pc_error` binary, and the
existing G-PCC artifact tree declared in the configuration. Outputs are written
under `artifacts/local/` and are not committed.

## Results

Placeholder. Do not infer compression or tail-flattening conclusions from the
implementation smoke. Report the full six-seed paired result, exact rates, and
negative or score-specific outcomes here after the predeclared run completes.
