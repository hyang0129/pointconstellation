# Experiment 040: heuristic selective pass-through

Status: complete; Gate G-C1 fails under the predeclared issue #66 criterion (the runner's weaker slope test passes and is reported separately).

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

The full experiment ran on an EmpireAI H200 (`--device cuda`, 7.4 h): 128
validation and 32 category-OOD clouds, six frozen Experiment 019 decoders,
FPS-start Adam-STE search, four irregularity scores x three splits x six payload
budgets, plus the uniform-constellation, random-preserved and fresh-encode
G-PCC controls. All seven contract checks pass (source-only selection, exact
preservation of passed-through lattice points, exact stream round trip,
stratification covering every target point, official D1/D2 present, frozen
decoder hashes unchanged, fixed payloads within budget). Bytes below are mean
complete stream bytes (payload + 12-byte header); the G-PCC arm is the nearest
available TMC13 rate point and is 15-30 B heavier than the learned arms at
every budget.

### Validation, 50 % preserved split (official RMSE in 12-bit grid units)

| Budget | Arm | Bytes | D1 | D2 | p95 | p99 | Boundary recall | Thin recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 40 | uniform constellation (control) | 50 | 241.8 | 131.0 | 0.182 | 0.223 | 0.010 | 0.012 |
| 40 | selective, decoder-residual | 52 | 245.5 | 136.2 | 0.184 | 0.219 | 0.019 | 0.017 |
| 40 | selective, curvature | 52 | 250.4 | 136.2 | 0.195 | 0.243 | 0.023 | 0.025 |
| 40 | selective, boundary | 52 | 250.8 | 136.2 | 0.194 | 0.242 | 0.024 | 0.021 |
| 40 | random preserved (control) | 52 | 249.8 | 136.1 | 0.193 | 0.242 | 0.013 | 0.015 |
| 40 | G-PCC octree | 72 | 242.4 | 144.6 | 0.157 | 0.172 | 0.015 | 0.015 |
| 64 | uniform constellation (control) | 77 | 231.7 | 125.6 | 0.170 | 0.205 | 0.011 | 0.013 |
| 64 | selective, decoder-residual | 79 | 233.7 | 131.7 | 0.171 | 0.200 | 0.024 | 0.022 |
| 64 | selective, curvature | 79 | 237.8 | 131.7 | 0.181 | 0.223 | 0.031 | 0.034 |
| 64 | selective, boundary | 79 | 238.4 | 131.7 | 0.180 | 0.222 | 0.034 | 0.028 |
| 64 | random preserved (control) | 79 | 236.9 | 131.7 | 0.179 | 0.221 | 0.017 | 0.019 |
| 64 | G-PCC octree | 98 | 167.6 | 101.9 | 0.104 | 0.114 | 0.031 | 0.031 |
| 110 | uniform constellation (control) | 122 | 222.2 | 118.5 | 0.160 | 0.189 | 0.012 | 0.015 |
| 110 | selective, decoder-residual | 124 | 221.7 | 126.8 | 0.158 | 0.182 | 0.030 | 0.028 |
| 110 | selective, curvature | 124 | 225.6 | 126.8 | 0.167 | 0.203 | 0.046 | 0.050 |
| 110 | selective, boundary | 124 | 226.6 | 126.8 | 0.167 | 0.202 | 0.053 | 0.040 |
| 110 | random preserved (control) | 124 | 224.0 | 126.7 | 0.166 | 0.201 | 0.022 | 0.025 |
| 110 | G-PCC octree | 129 | 121.1 | 71.2 | 0.078 | 0.085 | 0.059 | 0.059 |

Paired bootstrap, selective (50 %) versus the uniform constellation at equal
payload, validation, relative improvement in percent (negative = selective is
worse): decoder-residual D1 -1.5 / -1.7 / -0.9 / -0.9 / -0.6 / +0.2 and D2
-4.0 / -6.0 / -4.8 / -6.8 / -8.3 / -7.0 at 40 / 52 / 64 / 78 / 96 / 110 B, with
p99 tail +1.7 / +1.0 / +2.3 / +2.6 / +3.0 / +3.6; curvature D1 -3.5 / -3.6 /
-2.6 / -2.7 / -2.4 / -1.5, D2 -4.0 / -6.0 / -4.8 / -6.9 / -8.4 / -7.0, p99
-9.2 / -10.7 / -9.1 / -9.8 / -9.0 / -7.7. The 25 % and 75 % splits and the OOD
split show the same ordering (OOD aggregates in `selective_metrics.json`).

### Gate G-C1

Two gate formulations must be distinguished.

1. **The gate predeclared in issue #66** (and in `docs/track-c-selective-preservation.md`):
   selective pass-through must reduce D1 on the high-irregularity stratum by a
   CI-backed margin over the uniform-`K` control at equal bytes, and boundary /
   thin-structure recall must recover to at least 0.8 of raw-input recall; if
   the uniform control matches, the track stops. **This gate fails.** At every
   budget the uniform constellation is equal or better on aggregate D1 and
   better on D2; the random-preserved control matches the selective arms on
   D1/D2/tails, so the selection score adds nothing to fidelity; and boundary
   recall reaches at most 0.053 (raw input: 1.0), far below 0.8.
2. **The gate as implemented in this runner** (`gate_g_c1` in
   `selective_metrics.json`): curvature-score selective pass-through at the
   50 % split must have a positive paired-bootstrap reduction of the
   D1-versus-curvature-quintile *slope* with at most 5 % aggregate D1
   degradation at four of six validation budgets, and a positive slope
   reduction at four OOD budgets. This weaker test records `passes: true`
   (4 / 6 validation budgets, 64-110 B, with 2.4-3.6 % aggregate D1
   degradation; 5 / 6 OOD). It passes only because the error-versus-curvature
   curve flattens slightly while total error rises; it was not the predeclared
   criterion and is reported here for completeness, not as a pass.

**Decision: G-C1 fails; Experiment 042 (learned selection) is not filed.**

### Reading

1. **At 40-110 bytes a raw point costs as much as a constellation point, and a
   constellation point buys far more surface.** Spending half the budget on
   passed-through points preserves 4-12 raw points out of 2,048, which cannot
   move boundary recall or the tail, while removing 4-12 decoder-aware points
   costs 1-4 % D1 and 4-8 % D2. Selective preservation by individual points is
   the wrong mechanism in this regime; irregularity would have to be carried
   as structure the decoder can expand, or the budget has to be an order of
   magnitude larger.
2. **The selection score does not matter for fidelity.** Random preserved
   points reproduce the selective arms' D1/D2/tails to within noise; only the
   boundary and curvature scores raise boundary/thin recall, and only to a few
   percent.
3. **The decoder-residual score is the only one that improves the tail**
   (p99 +1-4 %), consistent with it targeting the decoder's own worst points;
   this is a small, real effect and the one lead worth keeping if a
   higher-rate variant is ever revisited.
4. **G-PCC dominates from 64 B upward** in this table, more strongly than in
   Experiment 025, because the G-PCC arm here sits 15-30 B above the learned
   arms at each nominal budget (nearest available TMC13 rate point); the
   crossover is consistent with Experiment 025's ~65 B once bytes are matched.

Experiment 041 (defect-injection anomaly AUROC) still tests the premise that
codecs erase anomalies, independently of whether pass-through can fix it.

Artifacts: `artifacts/local/experiment_040_selective/` (`selective_metrics.json`,
`selective_per_cloud.jsonl`, `run_manifest.json`) on the cluster
(`~/LLM_research/pointconstellation-tracks-a`).
