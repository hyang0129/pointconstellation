# Experiment 041: defect-injection anomaly benchmark

Status: implementation and real-codec CPU smoke complete. Full benchmark
results are pending selection of the Experiment 040 score/split cell and the
predeclared full run. The first full cluster attempt is invalid: defect
injection could leave the codecs' declared domain and stopped before producing
a complete result.

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
- `hole`: deletion of a local disk of finite samples;
- `thin_spur`: addition of a narrow, outward point sequence; and
- `surface_noise`: a local, randomly directed displacement patch.

The seed is the SHA-256-derived tuple `(defect_seed, cloud_id, defect_type)`, so
the result does not depend on dataloader or manifest traversal order. A defect
fraction is sampled in `[0.01, 0.05]` and rounded to an integer count that stays
inside those bounds. Dent, bump, and noise preserve `N`; a spur returns `N + K`
points; and a hole returns `N - K` points. The runner records requested and
realized fractions and the hole's removed count.

Every returned point has an aligned binary label. A deleted point cannot carry
a label, so a hole labels the equally sized nearest surviving rim and separately
records the deleted count. This is a point-task convention, not a claim that
the finite observed cloud samples the interior of an underlying continuous
hole.

The mesh loader's existing bounding-box-center, unit-maximum-radius transform
is the shared normalization for this benchmark. It correctly places each
undefected finite sample in the codecs' declared `[-1, 1]^3` domain. The failed
full-run attempt exposed a later protocol bug: bump, thin-spur, and surface-noise
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

This choice does not introduce a new per-cloud normalization or inverse
transform. The raw arm and every codec arm receive the exact same final
post-injection coordinate array, and the runner checks this path. The existing
shared normalized-domain codec contract and complete serialized-byte accounting
therefore remain unchanged; the injection factor describes construction of the
synthetic benchmark input and is not decoder side information. Point labels are
created and validated against the final returned array, so attenuation does not
change their alignment or the declared cardinality conventions.

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

The CPU smoke now uses these real providers rather than diagnostic subset
streams. It evaluates four fixture base clouds (two validation and two category
OOD), each as an undefected control and a deterministic dent, for eight codec
inputs total. It uses decoder seed 7, all six rates, all four coded arms, and the
three scorer seeds. This remains a plumbing and reproducibility check rather
than evidence for G-C2 or a compression result.

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

Placeholder. The real-codec smoke is a plumbing result only. Do not infer an
anomaly-preservation or compression conclusion until the full run evaluates the
predeclared selected cell, all three scorer seeds, and both ModelNet40 splits.
