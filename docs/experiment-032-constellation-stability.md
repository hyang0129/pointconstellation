# Experiment 032: constellation stability

Status: complete; Gate G-B3 fails — constellations are not repeatable point sets (resample repeatability < 3% at 16 bins), while decoded surfaces are stable and cross-decoder degradation stays at 15--27%.

## Question and hypothesis

Experiments 019--022 measure reconstruction quality but do not establish that
the transmitted coordinates have stable geometric meaning. Experiment 032
tests H-B3: are decoder-aware constellations repeatable across independent
surface samples and rigid poses, and do they retain useful meaning under a
different frozen decoder, or are they decoder-private codes?

The leading hypothesis concerns the Experiment 019 competitive refiner. A
keypoint-like result requires repeatability across independent samples and
translations, recovery under a pose-normalized PCA ablation, and bounded
quality loss when the exact coordinate stream is decoded by another stabilized
decoder seed. Raw rotation repeatability is expected to be poor because the
decoder and refiner are not rotation-equivariant. It is reported but is not a
pass requirement.

This experiment studies one fixed operating point. It is not a general
compression or rate-distortion claim.

## Fixed protocol

- Reuse the six sealed Experiment 019 stabilized decoders and the three
  refiner checkpoints assigned to each decoder. Check decoder hashes before
  and after evaluation.
- Evaluate all 128 validation and 32 category-OOD meshes. Each mesh supplies
  the deterministic, independent 2,048-point `source_points` and
  `fresh_points` samples already declared by the Experiment 019 manifest.
- Use `K=8`, 12-bit coordinates, and the canonical fixed-width bitstream. Every
  message has a 14-byte header and 36-byte coordinate payload, for exactly 50
  bytes. No matching, transform, PCA, score, decoder, or refiner metadata is in
  the message.
- Generate one seeded uniform rotation and one seeded translation per mesh.
  Translations are clipped only to the feasible interval that keeps both
  independent samples in `[-1, 1]^3`; coordinates are not clipped after the
  rigid transform.
- Encode each condition independently. The source sample for that condition is
  the only optimization target. The paired fresh sample is evaluation-only.
- Serialize and decode every constellation before calculating repeatability,
  Hausdorff distance, decoded consistency, or cross-decoder quality. Thus the
  measurements include exact lattice and canonical-order effects.

The four encoder arms are:

| Method | Fixed definition | Representation before serialization |
|---|---|---|
| `fps` | Experiment 021 deterministic mean-distant-start FPS | strict subset |
| `random_best_of_16` | Sixteen seeded random subsets, ranked by source-only decoder Chamfer | strict subset |
| `refiner` | Each assigned Experiment 019 competitive refiner | free coordinates |
| `adam_64` | Experiment 022 Adam/STE, FPS initialization, 64 source-only decoder evaluations | free coordinates |

The full config fixes Adam at 64 evaluations. The smoke config reduces this to
two evaluations and therefore cannot support an `adam_64` result; its method
label identifies the code path, while `decoder_evaluations` and the config
record the actual smoke budget.

## Repeatability conditions

For each encoder/decoder/refiner cell, four unordered comparisons are made:

1. `independent_sample`: encode the source and fresh mesh samples separately.
2. `translation`: encode the source and its translated copy, then apply the
   recorded inverse translation to the second serialized constellation and
   reconstruction.
3. `rotation`: encode the source and its rotated copy, then apply the recorded
   inverse rotation to the second serialized constellation and reconstruction.
4. `rotation_pca`: independently center and orient the source and rotated
   source with a sign-oriented, right-handed PCA frame before encoding.

The PCA condition is a pose-normalized ablation. The receiver reconstructs in
the canonical frame; it cannot recover the original pose from the coordinate
stream. PCA centers, axes, eigenvalues, and scale are written only to analysis
metadata so that the bookkeeping can be audited. They are not transmitted and
must not be counted as evidence for a pose-preserving coordinate-only codec.
Near-symmetric shapes can have unstable PCA axes, which is a legitimate failure
mode of this ablation.

## Metrics

Let the coordinate lattice step be

```text
delta = 2 / (2^q - 1).
```

At radius `r`, directed repeatability is the fraction of constellation points
whose Euclidean nearest-neighbor distance to the other unordered constellation
is at most `r * delta`. The reported repeatability is the mean of the two
directed fractions. The predeclared radii are 1, 4, 16, and 64 bins. This is a
finite-sample coordinate-set metric, not correspondence to the underlying
continuous surface.

The evaluator also reports symmetric Euclidean Hausdorff distance in coordinate
units and lattice bins, plus symmetric Chamfer RMSE between the two decoded
clouds after the same known-transform alignment.

For the cross-decoder test, a source constellation selected against decoder
seed A is serialized once and decoded by every seed B. Only those coordinates
transfer. Source and independent-sample Chamfer MSE are recorded for matched
`A -> A` and crossed `A -> B` cells. The primary degradation is the relative
change in independent-sample RMSE:

```text
100 * (crossed_fresh_RMSE - matched_fresh_RMSE) / matched_fresh_RMSE.
```

The analysis averages the complete decoder/refiner replicates within a mesh,
then uses a category-then-cloud bootstrap. Validation determines the gate;
category OOD is transfer evidence only.

## Gate G-B3

G-B3 applies to the leading refiner factorial at radius 16 bins. It passes only
if all four validation checks pass:

- the 95% confidence-interval lower bound for independent-sample
  repeatability is at least 0.50;
- the lower bound for translation repeatability is at least 0.75;
- the lower bound for pose-normalized PCA rotation repeatability is at least
  0.50; and
- the upper bound for cross-decoder independent-sample RMSE degradation is at
  most 50%.

Raw rotation repeatability is always reported as a finding but does not select
the pass decision. FPS, random-best-of-16, and Adam-64 are controls and receive
the same summaries, but they do not rescue a failed refiner gate.

A pass supports only the narrow statement that the refiner coordinates show
some repeatable and cross-decoder-stable structure at this fixed ModelNet40
operating point. A failure demotes the coordinates from keypoint-like geometry:
the paper must describe the relevant sample, pose, or decoder dependence and
must not imply that the learned coordinates are interchangeable geometric
landmarks. Cross-decoder failure specifically supports the decoder-private-code
explanation.

## Outputs

`constellation_pairs.jsonl` stores one row per mesh, encoder cell, and
repeatability condition. It includes both stream hashes and byte counts, exact
analysis transforms, PCA ablation metadata, repeatability at every radius,
Hausdorff distance, decoded consistency, encoding time, selection seeds, and
decoder-evaluation accounting.

`cross_decoder.jsonl` stores matched and crossed source-stream decodes with
actual byte counts, distortions, degradation, and decoded-cloud consistency.
`run_manifest.json` fixes data membership and hashes, model/checkpoint hashes,
device, seeds, and the complete config. `constellation_stability_metrics.json`
contains contract checks, bootstrap summaries, and G-B3.

The runner rejects an existing output directory whose run manifest differs. A
same-manifest rerun replaces the JSONL files after completing the factorial; it
does not treat partial files as resumable evidence.

## Results

Executed on an EmpireAI H200 (CPU-bound matching; 7.8 h): 23040 comparison rows and
34560 cross-decoder rows over the six sealed decoders, three refiner seeds, 128 validation and 32
category-OOD clouds; every message serialized as an actual 50-byte stream. Repeatability is the fraction of
constellation points re-found within `r` lattice bins (12-bit lattice) in the second encoding.

**Gate G-B3 fails** (thresholds at r=16: repeatability >= 0.5, translation repeatability >=
0.75, cross-decoder degradation <= 50.0%).

### Repeatability (validation; mean fraction of points re-found, 95% CI)

| Method | Resample r=16 | Resample r=64 | Translation r=16 | Translation r=64 | Rotation r=64 | Rotation+PCA r=64 |
|---|---|---|---|---|---|---|
| `fps` | 0.028 [0.011, 0.050] | 0.190 [0.136, 0.251] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| `random_best_of_16` | 0.001 [0.000, 0.002] | 0.015 [0.009, 0.024] | 0.005 [0.003, 0.008] | 0.015 [0.010, 0.023] | 0.016 [0.010, 0.024] | 0.018 [0.011, 0.029] |
| `refiner` | 0.005 [0.002, 0.009] | 0.096 [0.063, 0.133] | 0.056 [0.040, 0.076] | 0.365 [0.317, 0.412] | 0.026 [0.021, 0.032] | 1.000 [1.000, 1.000] |
| `adam_64` | 0.008 [0.004, 0.012] | 0.122 [0.089, 0.159] | 0.027 [0.018, 0.036] | 0.212 [0.181, 0.244] | 0.051 [0.039, 0.065] | 1.000 [1.000, 1.000] |

### Decoded-cloud consistency (validation; Chamfer RMSE between the two decodings)

| Method | Decoded consistency, resample | Decoded consistency, translation | Decoded consistency, rotation |
|---|---:|---:|---:|
| `fps` | 0.0302 | 0.0266 | 0.0972 |
| `random_best_of_16` | 0.0498 | 0.0539 | 0.0914 |
| `refiner` | 0.0387 | 0.0383 | 0.0980 |
| `adam_64` | 0.0348 | 0.0366 | 0.0963 |

### Cross-decoder transfer (fresh-resample RMSE degradation when decoded by a different decoder seed)

| Method | Val cross-decoder degradation | OOD cross-decoder degradation |
|---|---|---|
| `fps` | 2.6% [2.0, 3.2] | 3.6% [1.9, 5.6] |
| `random_best_of_16` | 16.6% [13.6, 19.7] | 15.2% [11.9, 18.5] |
| `refiner` | 15.1% [12.5, 17.9] | 15.5% [11.2, 19.6] |
| `adam_64` | 26.8% [23.2, 30.6] | 26.7% [20.2, 33.1] |

### Reading

1. **Decoder-aware constellations are not repeatable keypoints.** Re-encoding an independent sample of the same
   mesh re-finds essentially none of the same points within 16 bins for any method (FPS 2.8%, refiner 0.5%,
   Adam 0.8%), and only 10--19% within 64 bins. Even under a pure translation, which FPS handles exactly (100%),
   the refiner re-finds only 37% of its points within 64 bins and Adam 21%: the search lands in a different
   member of a large family of equivalent constellations each time.
2. **But the decoded surfaces are stable.** The two decodings of independent samples agree to Chamfer RMSE
   0.035--0.039 for the refiner and Adam (FPS 0.030, random best-of-16 0.050), i.e. within the reconstruction
   error itself. The constellation is a stable code *for the surface* without being a stable *point set*, which
   is consistent with Experiment 031 (surface-level gains) and Experiment 034 (only weak category-consistent
   placement).
3. **The code is decoder-tuned but not decoder-private.** Decoding a constellation with a different decoder seed
   degrades fresh-resample RMSE by 15% (refiner) and 27% (Adam-64) against 2.6% for FPS: free coordinates
   exploit decoder-specific structure but still transfer, so the message is geometry plus a moderate
   decoder-specific component, not an arbitrary private code.
4. **Rotation** without canonicalization destroys repeatability for every free-coordinate method (the decoder
   is not equivariant); PCA pose normalization restores it fully for FPS, the refiner, and Adam, and is therefore
   part of the deployable protocol. Random best-of-16 is the least stable arm under every condition.
5. For the paper: the constellation should be described as a compact surface code with many equivalent
   realizations, not as learned keypoints; H-B3 as stated ("repeatable like keypoints") is rejected, while the
   weaker property that matters for compression -- decoding stability -- holds.

Artifacts: `artifacts/local/experiment_032_constellation_stability/constellation_stability_metrics.json`; per-pair
and cross-decoder rows on EmpireAI under `pointconstellation-tracks-b/artifacts/local/experiment_032_constellation_stability/`.

## Reproduction

Run the one-cloud-per-split smoke with:

```bash
.venv-train/bin/python -m pointconstellation.stability_analysis \
  --config configs/experiment_032_constellation_stability_smoke.json \
  --device mps
```

Run the full predeclared evaluation with:

```bash
.venv-train/bin/python -m pointconstellation.stability_analysis \
  --config configs/experiment_032_constellation_stability.json \
  --device mps
```

Both commands require the ignored Experiment 019 checkpoints, the checked
ModelNet40 manifest and meshes, and sufficient time for the Adam-64 factorial.
They do not retrain a decoder or refiner.
