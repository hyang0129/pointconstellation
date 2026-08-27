# Experiment 033: encoder objective sweep

Status: `k8_n2048` regime complete; Gate G-B2 fails under its predeclared rules (single regime, classifier below 0.70, D1 regressions above 2%), while the measured trade-offs show the objectives are Pareto-distinct.
the stabilized decoder cells assigned to Experiment 038.

## Question and hypothesis

Experiment 033 tests H-B2: source Chamfer may be a poor per-cloud search
objective when the desired reconstruction must retain local surface orientation
or downstream semantic information. The experiment changes only the objective
used by the Experiment 022-style Adam/STE encoder search. It does not change the
coordinate-only message or train a decoder against evaluation clouds.

The comparison is descriptive until the full factorial, frozen classifier
quality control, and paired uncertainty gate below are complete. A smoke result
is not evidence for or against H-B2.

## Fixed protocol

- Evaluate `K in {4, 8, 16, 32}` and `N in {1024, 2048, 4096}` at `q=12`.
  Each cell names a sealed stabilized-decoder config and artifact directory.
- Reuse the same decoder checkpoints, source clouds, deterministic
  initializers, Adam learning rate, and decoder-evaluation budget across the
  four objectives. Decoder and feature-extractor parameters remain frozen.
- Canonically sort each source set before initialization and derive any
  selector seed from that sorted source tensor plus global method settings.
  Split names, model identifiers, categories, and labels do not affect the
  encoder initialization.
- Serialize every selected constellation through the canonical bitstream. The
  complete per-cloud learned message remains one unordered, quantized `K x 3`
  coordinate set. Labels, classifier features, and normals are not serialized.
- Run official MPEG `pc_error` D1/D2 after encoding. Also report source and
  fresh-resampling Chamfer, finite-resampling D1/D2 proxies, p95, p99,
  Hausdorff, sliced-Wasserstein proxy, encode time, and classifier accuracy.
- Treat nearest distance to the finite source or fresh cloud as a sample
  proxy. `independent_finite_mesh_resample_proxy` is not labeled distance to a
  continuous surface.

The checked-in full config contains all 12 cells. The existing Experiment 019
artifact supplies only `K=8, N=2048`; the other paths intentionally name the
pending Experiment 038 dependencies. The runner stops at the first missing
config or artifact rather than silently dropping a regime.

## Objective registry

The registry exposes four scorers. Every scorer returns one differentiable
loss per cloud and receives an `ObjectiveContext` that structurally excludes
labels, fresh samples, target-only points, and provided normals.

1. `chamfer`: symmetric squared Chamfer between the frozen-decoder
   reconstruction and encoder-visible source cloud.
2. `point_to_plane`: symmetric squared point-to-plane loss. Unoriented normals
   are estimated once from the source cloud alone using deterministic PCA over
   source k-nearest-neighbor neighborhoods. Squared projections make normal
   orientation irrelevant. Analytic or mesh normals are evaluation-only.
3. `feature_matching`: mean squared distance between frozen PointNet
   penultimate features of the reconstruction and source.
4. `mixed`: `lambda * Chamfer + (1 - lambda) * feature_matching`, with
   `lambda=0.5` in the predeclared configs.

The PointNet classifier uses shared pointwise layers plus mean/max pooling, so
its feature and logits are invariant to point order. It is trained with labels
from the selected regime's training partition only. At encode time it is a
fixed function of source or reconstructed coordinates; no label is passed to
the objective. Labels are read later to score source-cloud and reconstruction
accuracy. Evaluation labels absent from the training class vocabulary are
recorded as unknown and excluded from accuracy instead of being counted as an
arbitrary error.

If no classifier checkpoint is configured, the runner deterministically trains
one and saves it under the output directory. A resume verifies the checkpoint
file hash, model-state hash, architecture metadata, and training-membership
hash. A supplied checkpoint can additionally be pinned with
`expected_classifier_sha256`.

## Outputs and checks

`objective_sweep_per_cloud.jsonl` contains one source-selected search result for
every regime, split, decoder seed, cloud, and objective. Each row includes:

- the stream bytes as hex, SHA-256, actual bits and bits per input point;
- objective value, selected start, decoder evaluations, and encode time;
- source/fresh Chamfer, official D1/D2, and the available finite-resampling and
  tail geometry proxies;
- source and reconstruction classifier correctness where the label is known;
  and
- explicit source-only, post-encode label/normal use, exact lattice, and
  bitstream round-trip fields.

`run_manifest.json` pins the normalized config, execution device, `pc_error`,
classifier, stability config/metric hashes, and dataset memberships. A changed
manifest cannot resume into an existing output. Identical streams reuse
official metrics; complete per-cloud rows are not appended twice. Decoder and
classifier state hashes are checked after search.

`objective_sweep_metrics.json` contains distortion-versus-accuracy Pareto
tables for every regime and split. D1 and D2 aggregates are computed as RMSE
from official MSE values. The Pareto flag uses lower official D1 RMSE and higher
reconstruction accuracy. It is undefined when a split contains no label in the
training vocabulary.

The dedicated surface-measurement pipeline from issue #47 and the downstream
pipeline from issue #46 are not present in this worktree. This implementation
provides the frozen training-only PointNet evaluation required for the sweep and
reports the geometry metrics already supported by the repository. It does not
rename those proxies as the missing issue #47 measurements. Those fields can be
added after that pipeline merges without changing the encoder objectives.

## Gate G-B2

Validation selects the conclusion; OOD rows are transfer diagnostics and
cannot rescue a failed validation gate. Each non-Chamfer objective is compared
with its matched Chamfer rows using a paired decoder-cloud-cell bootstrap. This
bootstrap is not presented as the repository's hierarchical category/cloud
analysis; the full paper table should also receive that analysis before a
headline claim.

A candidate passes one regime only when all of the following hold:

- the frozen classifier's source-cloud accuracy is at least 70%;
- the lower bound of the 95% paired interval for reconstruction-accuracy
  improvement is greater than zero; and
- the upper bounds of the 95% paired intervals for official D1 and D2 RMSE
  regression are each at most 2%.

G-B2 passes only if the same non-Chamfer objective passes at least two regimes
that collectively span at least two `K` values and at least two `N` values. A
pass supports further study of that encoder objective; it does not by itself
show better rate-distortion performance, general downstream-task preservation,
or continuous-surface fidelity. A failure keeps Chamfer as the primary search
objective and preserves the negative objective result.

## Reproduction

Run the fixture-scale configuration after producing the Experiment 019 smoke
artifact and building `pc_error`:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.objective_sweep_experiment \
  --config configs/experiment_033_objective_sweep_smoke.json \
  --device cpu
```

Run the full factorial after all Experiment 038 stabilized dependencies exist:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.objective_sweep_experiment \
  --config configs/experiment_033_objective_sweep.json \
  --device mps
```

Generated classifiers, streams, metric scratch files, and results stay under
`artifacts/local/` and are not committed.

## Results

Executed on an EmpireAI H200 for the one regime with sealed stabilized decoders (`k8_n2048`, six decoders,
Adam-STE search, 3840 rows). The other eleven `K x N` regimes require the Experiment 038 decoders,
which do not exist yet, so the factorial is incomplete by construction. The frozen PointNet feature extractor
was trained on training-split clouds only and reaches 0.547 top-1 on validation *source* clouds.

**Gate G-B2 fails** under its predeclared rules (it requires at least two regimes spanning two `K` and two `N`,
a source-cloud classifier accuracy of at least 0.70, and a D1/D2 regression of at most 2%); none of the three
conditions is met. The measured trade-offs are nonetheless decisive about the hypothesis:

| Candidate vs Chamfer | Accuracy delta (95% CI) | D1 regression | D2 regression | Passes |
|---|---|---|---|---|
| `point_to_plane` | +0.020 [0.003, 0.036] | +12.5% [10.8, 14.3] | -1.9% [-3.9, 0.4] | False |
| `feature_matching` | +0.202 [0.172, 0.232] | +68.2% [63.7, 72.7] | +122.5% [106.2, 138.5] | False |
| `mixed` | +0.198 [0.169, 0.229] | +58.1% [54.6, 61.6] | +99.8% [86.9, 112.1] | False |

| Split | Objective | D1 RMSE | D2 RMSE | Chamfer RMSE | Recon. classifier acc. | Encode s |
|---|---|---:|---:|---:|---:|---:|
| validation | `chamfer` | 239.2 | 128.2 | 0.0950 | 0.066 | 0.247 |
| validation | `point_to_plane` | 269.0 | 125.7 | 0.1050 | 0.086 | 0.217 |
| validation | `feature_matching` | 402.3 | 285.2 | 0.1625 | 0.268 | 0.112 |
| validation | `mixed` | 378.2 | 256.0 | 0.1534 | 0.264 | 0.262 |
| ood | `chamfer` | 248.2 | 139.2 | 0.1011 | n/a (OOD labels unseen) | 0.250 |
| ood | `point_to_plane` | 290.6 | 133.1 | 0.1136 | n/a (OOD labels unseen) | 0.220 |
| ood | `feature_matching` | 430.6 | 328.5 | 0.1739 | n/a (OOD labels unseen) | 0.115 |
| ood | `mixed` | 396.4 | 285.2 | 0.1623 | n/a (OOD labels unseen) | 0.265 |

### Reading

1. **The objectives are Pareto-distinct, so PSNR and task utility genuinely trade off at this rate.** Optimizing
   the eight coordinates to match the frozen classifier's features of the reconstruction makes the decoded cloud
   four times more recognizable (6.6% -> 26.8% top-1; +20 points, CI [17, 23]) while raising D1 by 68% and D2
   by 122%. The mixed objective sits between the two (26.4%, +58% D1).
2. **Point-to-plane is the interesting cheap alternative**: +2.0 points accuracy (CI [0.3, 3.6]) and slightly
   *better* D2 (-1.9%, CI [-3.9, 0.4]) for 12.5% worse D1. With D2 as the primary metric, point-to-plane would be
   the better encoder objective at no utility cost.
3. **Reconstructions from eight Chamfer-optimized points are nearly unrecognizable** to a classifier trained on
   real clouds (6.6% top-1 against 54.7% on the source). This is the decoder-side counterpart of Experiment 030's
   finding that raw constellations outclass decoded clouds as task inputs: geometric fidelity as measured by
   Chamfer/D1 and recognizability are different targets, and the shared decoder is trained only for the first.
4. Encode cost is objective-dependent (feature matching is the cheapest, 0.11 s; Chamfer 0.25 s per cloud on H200
   at 64 evaluations), which matters for the amortization story of Experiment 022.
5. To pass G-B2 as written, the missing regimes (Experiment 038 decoders for `K in {4, 16, 32}`,
   `N in {1024, 4096}`) and a stronger frozen classifier (>= 0.70 source accuracy, i.e. more training meshes
   or a larger backbone) are required; both are decoder/data investments, not encoder changes.

Artifacts: `artifacts/local/experiment_033_objective_sweep_k8/objective_sweep_metrics.json`,
`objective_sweep_per_cloud.jsonl`, `pointnet_classifier.pt` (cluster: `~/LLM_research/pointconstellation-tracks-a`).

