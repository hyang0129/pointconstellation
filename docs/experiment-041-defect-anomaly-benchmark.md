# Experiment 041: defect-injection anomaly benchmark

Status: complete; Gate G-C2 fails (selective pass-through does not recover cloud-level anomaly AUROC; the learned codec's decodes are at chance while G-PCC preserves the signal; the non-learned detector is under-powered).

## Question and hypothesis

Experiment 041 tests H-C1 on a task where the anomaly is the label: does a
codec erase the local geometry needed by an anomaly detector, and does
selective pass-through recover it at matched serialized bytes?

The anomaly scorer is fitted before codec evaluation. It receives only
undefected raw training coordinates. It is never fitted or calibrated on a
decoded cloud and never receives an encoder, decoder, codec stream, target
resample, mesh, normal, category, or defect label. Codec calls expose only the
following interface:

```text
encode(source_coordinates) -> complete_stream_bytes
decode(complete_stream_bytes) -> reconstructed_coordinates
```

Experiment 041 does not make the selective message a strict coordinate-only
constellation. As in Experiment 040, selective pass-through is an explicitly
labeled two-set coordinate ablation. The constellation-only arm remains the
strict unordered coordinate-set message.

## Data and deterministic defects

The full configuration reuses the official ModelNet40 stability manifest and
sampling seed. Undefected training clouds fit the normal manifold. Evaluation
uses the manifest's validation and category-OOD splits. Each evaluation mesh
has one exact undefected copy and one independently seeded instance of every
defect:

- `dent`: a cosine-tapered local inward displacement;
- `bump`: the corresponding outward displacement;
- `hole`: removal of a local disk followed by deterministic resampling from the
  surviving finite samples;
- `thin_spur`: relocation of a local patch into a narrow, outward point
  sequence; and
- `surface_noise`: a local, randomly directed displacement patch.

The seed is the SHA-256-derived tuple `(defect_seed, cloud_id, defect_type)`, so
the result does not depend on dataloader or manifest traversal order. A defect
fraction is sampled in `[0.01, 0.05]` and rounded to an integer count that stays
inside those bounds. The declared regime is exactly `N=2048` source points for
the checked full and smoke configurations. Every control and defect returns
exactly that `N`: dent, bump, and noise displace selected source samples; a spur
relocates its selected local patch rather than appending samples; and a hole
removes its patch and restores `N` by deterministically resampling points from
the surviving empirical cloud. The hole resamples existing coordinates without
jitter, so it does not invent a continuous surface. The runner records the
requested and realized fractions and the hole's removed count; under this
policy the replacement count equals that removed count.

Every returned point has an aligned binary label of length `N`. A deleted point
cannot carry a label, so a hole labels the equally sized nearest surviving rim,
labels its resampled duplicates as normal, and separately records the deleted
count. This is a point-task convention, not a claim that the finite observed
cloud samples the interior of an underlying continuous hole.

The cardinality policy is a hard protocol constraint, not a decode-time repair.
`DefectResult` rejects every injection whose returned coordinate or label count
differs from its source count; the benchmark additionally rejects any condition
that differs from configured `num_points`; and the real Experiment 040 provider
rejects any source that differs from that declared regime. There is no
variable-cardinality configuration and no silent source or decode resampling.
A future defect type that lacks a cardinality-preserving construction must
remain absent from the config-level `defect_types` set until it has one.

The mesh loader's existing bounding-box-center, unit-maximum-radius transform
is the shared normalization for this benchmark. It correctly places each
undefected finite sample in the codecs' declared `[-1, 1]^3` domain. The first
failed full-run attempt exposed a protocol bug: bump, thin-spur, and surface-noise
generation (and, depending on local normal direction, a dent) could displace a
valid normalized sample beyond that domain. The interior fixture smoke did not
exercise this boundary case.

Experiment 041 now preserves the fixed domain during injection. It first
asserts that the source lies in `[-1, 1]^3`. For every displacement defect, it
forms the originally declared displacement field and multiplies the complete
field by the largest single factor in `(0, 1]` that keeps every returned
coordinate in the cube. The one factor preserves directions and relative
taper, length, and radius instead of clipping individual coordinates. Hole and
control conditions use factor one. An injection that cannot realize any
nonzero displacement in the domain fails with an explicit error. The realized
`domain_scale_factor`, number of attenuated conditions, and minimum factor are
written to the machine-readable rows and data protocol so boundary attenuation
cannot be hidden when interpreting defect-stratified results.

These choices do not introduce a new per-cloud normalization or inverse
transform. The raw arm and every codec arm receive the exact same final
post-injection coordinate array, and the runner checks this path. The existing
shared normalized-domain codec contract and complete serialized-byte accounting
therefore remain unchanged; the injection factor describes construction of the
synthetic benchmark input and is not decoder side information. Point labels are
created and validated against the final returned `N`-point array, so attenuation
does not change their alignment or the declared cardinality convention. The
machine-readable data protocol records both the domain policy and the
fixed-cardinality construction.

Decoded labels use nearest-defected-target-point transfer when output
cardinality differs. This is explicitly a nearest distance to a finite sampled
target, not a continuous-surface correspondence.

The checked external-manifest hook accepts version-1 `MVTec 3D-AD` and
`Real3D-AD` manifests. It validates records and hashes declared by the manifest
consumer but performs no download. Each record declares `model_id`, `category`,
`pointcloud`, `pointcloud_sha256`, and a binary `cloud_label`; point labels may
be added by the eventual dataset-specific loader. External data are not part of
the current gate.

## Non-learned anomaly scorer

The first benchmark scorer is a k-nearest distance-to-normal-manifold baseline.
It retains a seeded, bounded subset of raw normal training clouds and points.
A coordinate distribution descriptor chooses nearby normal training shapes.
For each candidate shape, point scores combine:

1. decoded-to-normal nearest-sample distance; and
2. normal-to-decoded nearest-sample distance attributed to its nearest decoded
   point.

The reverse term makes missing finite geometry observable rather than assigning
zero anomaly to every point that remains after a hole. The cloud score is the
mean of the largest 2% of point scores. The full experiment repeats the bounded
reference subsampling with seeds 4101, 4111, and 4127. These are scorer seeds,
not codec or defect seeds. A learned PointNet++/DGCNN scorer remains a subsequent
extension; it must use the identical raw-only fit boundary and cannot replace
this predeclared non-learned baseline after results are inspected.

## Arms and rate accounting

The checked provider returns the following arms at all six Experiment 040
coordinate-payload budgets: 40, 52, 64, 78, 96, and 110 bytes.

- raw input, evaluated without a coded rate as an upper bound;
- constellation-only;
- selective pass-through using the Experiment 040-selected score and `K1/K2`
  split;
- the same `K1/K2` split with coordinate-keyed random `K2`; and
- G-PCC at the nearest measured actual complete stream size.

`CodecArm` records the common coordinate-payload budget and the complete target
stream bytes separately. Exact learned/selective arms must return nonempty byte
strings whose measured lengths equal their complete targets. At least two exact
arms at a target are checked for equal actual serialized length on every cloud.
Any baseline padding or framing used for equality is part of the byte string and
is therefore counted. G-PCC may declare a nonzero maximum rate error; its actual
complete stream length remains in every per-cloud row and is not described as
an exact match.

The real provider is
`pointconstellation.exp040_defect_codecs:build_codec_arms`. It loads and checks
the selected sealed Experiment 019 decoder, then uses Experiment 040's
deterministic FPS start and Adam-STE source-Chamfer search for every required
`K` or `K1`. The ordinary constellation stream has a 14-byte header, so its
complete stream appends two counted zero framing bytes to match the selective
16-byte header. The resulting complete learned-arm targets are 52, 66, 79, 93,
111, and 124 bytes. Decoding validates and removes those two bytes before
passing only the serialized constellation coordinates to the frozen decoder.
Every provider encoder first requires exactly configured `num_points` source
coordinates. Both the Adam-STE scoring decode and the serialized learned stream
request exactly that configured output count, and the provider rejects a regime
above the sealed decoder maximum. It never derives the request from a defected
cloud's post-injection length. This makes a cardinality defect fail at the
benchmark/provider boundary rather than inside the decoder and does not raise
the sealed model's maximum.

Selective and random-`K2` arms use the same `K1/K2` at each rate. The selected
source-only score and preserved fraction come directly from
`selective_score_method` and `selective_preserved_fraction` in the Experiment
041 config; they are never chosen from anomaly results. Until the full
Experiment 040 result fixes its best cell, both configs default to
`decoder_residual` at a 50% preserved split. Updating those two fields is the
only selection change needed once that result exists.

G-PCC reuses Experiment 040's TMC13 argument grid but freshly encodes each
Experiment 041 source. A shared per-source frontier is matched by nearest actual
geometry-brick payload at each ladder point. Every row declares the selected
TMC13 rate point, parsed header and payload bytes, signed payload-budget delta,
complete-stream target delta, and complete stream hash. The checked configs set
a 512-byte maximum complete-stream mismatch; the run fails if it is exceeded.

The provider is supplied as `module:function`. It is called as
`function(config=config, device_name=device_name)` and returns `CodecArm`
instances. This narrow hook keeps the benchmark blind to encoder internals: its
only calls remain `encode(source_coordinates)` and `decode(stream)`. Defect
labels are retained by the evaluation layer for metric computation and never
enter either call. The runner rejects a real provider that omits any
constellation, selective, random-`K2`, or G-PCC ladder cell.

The CPU smoke uses these real providers rather than diagnostic subset streams.
It evaluates four fixture base clouds (two validation and two category OOD) at
the full `N=2048` regime, each as an undefected control, a deterministic bump,
and a deterministic thin spur, for twelve codec inputs total. It uses the real
sealed decoder seed 7 when the checked artifact symlink is available, all six
rates, all four coded arms, and the three scorer seeds. This remains a plumbing
and reproducibility check rather than evidence for G-C2 or a compression result.

## Metrics and uncertainty

The runner reports:

- cloud AUROC and AUPRC from the per-cloud tail score;
- point AUROC and AUPRC for each defected output where transferred labels
  contain both classes;
- summaries stratified by defect type and by realized size (`1-2%`, `2-4%`,
  and `4-5%`); and
- measured output cardinality, defective-label count, complete stream bytes,
  and stream SHA-256 per cloud.

Intervals resample scorer seeds and base-cloud identities hierarchically. Arm
comparisons keep scorer, cloud, and defect condition paired. Point metrics first
compute AUROC/AUPRC within each eligible defected cloud, then average across
clouds and scorer seeds so large decoded point sets do not silently receive more
weight. A missing point metric is reported as unavailable rather than imputed.

`defect_anomaly_metrics.json`, `defect_per_cloud.jsonl`, and
`run_manifest.json` contain the complete configuration, dataset membership and
manifest hash, scorer seeds, codec roles, actual rates, summaries, gate inputs,
contract checks, and artifact hashes.

## Resume, cache, and progress protocol

The runner writes `run_manifest.json` before data loading or codec work. Its run
identity hashes the resolved Experiment 041 configuration, Python source tree,
device, dataset manifest, Experiment 040 and stability configurations, G-PCC
reference grid and executable, sealed decoder checkpoint and selection, and it
records the defect seed explicitly. Existing scored rows are resumed only when
that complete identity matches. A missing, unreadable, or mismatched manifest
starts a clean per-cloud result file and emits the reason; rows from different
configurations, code, models, dependencies, or devices are never combined.

`defect_per_cloud.jsonl` is append-only during evaluation. Each scorer-seed row
is flushed and synced immediately after one split/cloud/condition/arm/rate
decode has been scored. A restart validates that existing rows form the
canonical completed prefix, fits the raw-only scorers before any remaining
codec call, skips complete units, and appends only missing rows. A partial final
line from abrupt termination is discarded. The final JSONL therefore has the
same bytes as an uninterrupted run.

The expensive provider intermediates live below `codec_scratch/`. Adam-STE
coordinates are keyed by canonical source-cloud SHA-256 plus search parameters,
decoder identity, Experiment 040 configuration, and code hash. Every cached
array has a checked content hash. Each G-PCC frontier cell is likewise keyed by
source hash, TMC13 rate-point arguments, position precision, executable hash,
and code hash; its stream and reconstruction hashes are checked before reuse.
Missing, partial, mismatched, or corrupted entries are recomputed and replaced.
Cache entries are not treated as per-cloud messages and do not alter rate
accounting or decoded coordinates.

During execution, stdout receives compact JSON progress lines containing
`stage`, `done`, `total`, and `elapsed_seconds`, with the current condition and
arm during evaluation. The pre-existing final human-facing summary remains the
last output. `defect_anomaly_metrics.json` also reports wall-clock time for
defect injection, scorer fitting, encode and decode aggregated by arm, anomaly
scorer/metric computation (`official_metrics_seconds`), hierarchical bootstrap,
and unattributed orchestration. G-PCC's internally paired TMC13 encode/decode
subprocesses occur behind its `encode(source)` frontier call and are therefore
charged to the G-PCC encode arm; the benchmark-visible `decode(stream)` lookup
is reported separately.

## Predeclared gate G-C2

The primary operating point is the 64-byte Experiment 040 coordinate-payload
budget on ModelNet40 validation. Before the full run, the Experiment 040
selection artifact fixes the selective score and `K1/K2` split. Experiment 041
does not reselect them from anomaly results.

G-C2 passes only if the selective arm has a strictly positive lower 95% paired
hierarchical-bootstrap bound against both constellation-only and random-`K2`
for both:

1. cloud AUROC across all five defect conditions and their paired controls; and
2. mean within-cloud point AUROC across eligible defected outputs.

All four lower bounds must exceed zero. Validation AUPRC, the other five rates,
defect/size strata, raw input, G-PCC, and category OOD are reported controls and
cannot rescue a failed gate. A pass supports selective recovery only for the
tested scorer, selected Experiment 040 policy, rate, dataset, and defects. A
failure does not imply that every codec erases every anomaly.

## Commands

Run the checked eight-cloud, one-decoder fixture smoke. It writes under
`artifacts/local/` by default; use `--output-dir` to redirect it:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.defect_anomaly_benchmark \
  --config configs/experiment_041_defect_anomaly_smoke.json \
  --device cpu
```

Run the full benchmark after setting the Experiment 040-selected score and
preserved fraction in the full config:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.defect_anomaly_benchmark \
  --config configs/experiment_041_defect_anomaly.json \
  --device mps
```

`--device cuda` and `--device cpu` are also accepted. `--codec-provider` can
still override the checked config hook for a controlled provider test. The full
configuration requires the ignored official ModelNet40 manifest/data, frozen
Experiment 019 decoder checkpoints, the completed Experiment 040 selection,
and the pinned G-PCC executable. Neither command launches training.

## Results

The full benchmark ran on homen-linux (RTX 5070 Ti, `--device cuda`, 75 min
with the vectorized scorer; the first cluster attempt was killed by an
allocation time limit and a second attempt spent 21 h in the pre-vectorization
bootstrap). 128 validation and 32 category-OOD clouds, one undefected control
plus five defect conditions per cloud, six payload budgets, the four codec arms
plus raw input, three scorer seeds, 72,000 scored rows. All thirteen contract
checks pass (scorer fitted on undefected raw training clouds only; encoders
receive coordinates only; every injected cloud inside the codec domain and at
the declared cardinality; exact arms at target bytes; all codec inputs equal
the raw-arm input). Selective arm: decoder-residual score, 50 % preserved
split (the Experiment 040 default; Experiment 040 found no score that beats
the random control on fidelity).

### Overall anomaly detection, validation (mean over three scorer seeds)

| Arm | Bytes | Cloud AUROC | Cloud AUPRC | Point AUROC | Point AUPRC |
|---|---:|---:|---:|---:|---:|
| raw input (upper bound) | - | 0.576 | 0.865 | 0.658 | 0.198 |
| constellation-only | 79 | 0.501 | 0.836 | 0.606 | 0.159 |
| selective pass-through | 79 | 0.505 | 0.840 | 0.657 | 0.170 |
| random-K2 preserved (control) | 79 | 0.494 | 0.835 | 0.645 | 0.158 |
| G-PCC octree (nearest rate point) | 79 | 0.602 | 0.872 | 0.598 | 0.238 |

Across the 40-110 B ladder the ordering does not change: constellation-only
cloud AUROC stays at 0.49-0.51, selective at 0.50-0.52, random-K2 at
0.49-0.52, G-PCC at 0.58-0.60. Category-OOD at 64 B: raw 0.567 / 0.608,
constellation-only 0.494 / 0.622, selective 0.500 / 0.646, random-K2
0.497 / 0.637, G-PCC 0.585 / 0.582 (cloud / point AUROC).

Per defect type at 64 B (cloud / point AUROC): raw dent 0.57 / 0.65, bump
0.56 / 0.66, hole 0.50 / 0.53, thin spur 0.70 / 0.75, surface noise
0.54 / 0.69; constellation-only 0.49-0.52 cloud AUROC for every type
(thin spur 0.52 / 0.69); selective 0.49-0.53 (thin spur 0.50 / 0.73); G-PCC
dent 0.60, bump 0.60, thin spur 0.70, surface noise 0.60, hole 0.51. By
defect size (cloud AUROC, raw -> constellation-only -> G-PCC): small 1-2 %
0.54 -> 0.51 -> 0.57, medium 2-4 % 0.57 -> 0.49 -> 0.60, large 4-5 %
0.63 -> 0.49 -> 0.64.

### Gate G-C2

Predeclared rule: selective pass-through must beat both the constellation-only
and the random-K2 baselines on cloud and point AUROC with paired hierarchical
bootstrap lower bounds above zero at the primary cell (validation, 64 B).

| Comparison | Metric | Difference | 95 % CI | Positive |
|---|---|---:|---|---|
| selective - constellation-only | cloud AUROC | +0.004 | [-0.022, +0.032] | no |
| selective - constellation-only | point AUROC | +0.051 | [+0.031, +0.072] | yes |
| selective - random-K2 | cloud AUROC | +0.011 | [-0.006, +0.028] | no |
| selective - random-K2 | point AUROC | +0.012 | [+0.006, +0.017] | yes |

**G-C2 fails.** Selective pass-through does not recover cloud-level anomaly
detection, and its point-level gain is reproduced by random preserved points
(the +0.012 residual over random is real but an order of magnitude below the
raw-versus-decode gap in the other direction). The epic's stop rule (Track C
design note) also asked whether the premise itself holds: it does, weakly.
Constellation-only decodes sit 7.5 cloud-AUROC points below raw (0.501 vs
0.576) and at chance for every defect type and size, so the learned codec does
erase whatever anomaly signal the detector can see, and G-PCC (which keeps
points) does not. But the detector's ceiling on raw input is itself only
0.576 cloud AUROC, so the benchmark as built cannot resolve differences among
the learned arms; it is under-powered, not decisive.

### Reading

1. **The learned codec erases the anomaly signal; G-PCC preserves it.** At
   every budget and for every defect type, constellation decodes are at chance
   for cloud-level detection while G-PCC decodes match or exceed raw. This is
   the structured-loss premise of Track C, observed directly, and it is the
   most useful number this experiment produced.
2. **Pass-through of a handful of raw points does not restore it.** Preserving
   4-12 points of 2,048 lifts point AUROC because those points are themselves
   irregular and get flagged, which random points reproduce; it does not make
   the decoded *cloud* look anomalous. Combined with Experiment 040 this closes
   selective point pass-through as the mechanism at 40-110 B.
3. **The detector is the limiting factor.** A k-NN distance-to-normal-manifold
   scorer reaches only 0.50-0.70 cloud AUROC on raw defected clouds (holes are
   undetectable by construction: removing points leaves no off-manifold
   points). The learned scorer (PointNet++/DGCNN per-point, stage 2 of #67)
   and larger or real defects (MVTec 3D-AD / Real3D-AD, manifest hook present)
   are required before "codecs erase anomalies" can be quantified precisely
   enough to compare mechanisms; reading 1 is robust to this because it
   compares each arm against the same detector.

Timing (this run): encode 1,587 s constellation-only + 1,301 s selective +
282 s G-PCC + 19 s random-K2; scorer inference 177 s; label transfer 28 s;
point metrics 15 s. Artifacts:
`artifacts/local/experiment_041_defect_anomaly/` on homen-linux
(`defect_anomaly_metrics.json`, `defect_per_cloud.jsonl`, `run_manifest.json`).
