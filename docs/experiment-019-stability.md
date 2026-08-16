# Experiment 019: decoder/refiner stability decomposition

Status: implemented and executed locally under
[issue #13](https://github.com/hyang0129/pointconstellation/issues/13).

## Outcome

The stability and internal learned-codec gates pass on the declared ModelNet40
slice. Across six decoder seeds crossed with three common refiner seeds, the
bundled longer-training, EMA, and calibration-selection protocol reduces the
validation decoder-marginal Q90 Chamfer RMSE by 42.73% (95% decoder bootstrap
CI 39.61-44.86) and category-OOD Q90 by 38.98% (34.58-40.79). Every decoder
seed improves. The stabilized refiner retains a 17.26% median validation and
16.94% category-OOD advantage over FPS through the same decoder.

Against the frozen Experiment 018 feature codec at exactly 50 serialized bytes,
the stabilized constellation improves validation Chamfer RMSE by 29.90% (95%
independent-model, hierarchical category/cloud bootstrap CI 23.26-35.77) and
category OOD by 29.01% (13.14-41.36). Fresh independent surface resampling gives
the same conclusion. This resolves Experiment 018's internal fixed-width
feature-baseline failure, but it is not yet a published entropy-coded SOTA
comparison.

The variance diagnosis is decisive: decoder seed explains 99.998% of the
nonnegative baseline validation log-RMSE variance. After stabilization it still
explains 99.15%, but the decoder component itself falls from 0.01969 to 0.000532
(a 97.3% reduction). The bootstrap lower bound for the decoder-to-refiner-plus-
interaction variance ratio is 16,133 before stabilization and 18.53 after it,
so decoder training—not refiner initialization—is the dominant instability in
this design.

These results support continuing the Experiment 005 pathway. They do not yet
establish superiority over published learned codecs, official D1/D2 after
stabilization, a full ModelNet40 test, or a common MPEG/JPEG test condition.

## Question

Experiments 017-018 established that the competitive recurrent refiner improves
over FPS through the same decoder, but did not establish a reliable absolute
advantage over an exactly byte-matched learned feature codec. The weak
constellation seed still beat its own FPS control, which leaves decoder quality,
refiner initialization, and decoder/refiner interaction as competing causes.

Experiment 019 separates those sources of variation and evaluates one bundled
stabilization protocol before any larger architecture or dataset sweep.

## Predeclared hypotheses

1. Decoder seed accounts for more absolute-quality variance than refiner seed
   plus decoder/refiner interaction.
2. A source-visible calibration score predicts held-out decoder quality.
3. Longer decoder training, EMA, and calibration-based checkpoint selection
   raise the bad-seed floor without removing the FPS-relative refiner gain.

The third arm is a bundled intervention. A positive result does not identify
which of duration, EMA, or checkpoint selection caused the improvement.

## Fixed representation and data contract

- The complete learned message is an unordered, exact-quantized `K x 3`
  coordinate set.
- Decoder-gradient feedback may inspect only the encoder-visible source cloud.
- Decoder weights remain frozen and hash-identical during each refiner run.
- Training preserves Experiment 017's `K={4,8,16,32}` message curriculum;
  `N=2048,K=8,q=12` is the primary evaluation point.
- The 512 training, 128 official-test validation, and 32 category-OOD records
  are identical to Experiment 017.
- Calibration contains the next four deterministically ranked official-training
  meshes from each of the 32 training categories. It excludes the eight held-out
  categories and is disjoint from every other split.
- Validation and category-OOD metrics cannot affect candidate, arm, decoder,
  refiner, checkpoint, or hyperparameter selection.

Regenerate the ignored local manifest after accepting ModelNet40's terms:

```bash
.venv/bin/python -m pointconstellation.mesh_manifest \
  --dataset modelnet40 \
  --root data/modelnet40_official/ModelNet40 \
  --output configs/manifests/modelnet40_stability.local.json \
  --archive data/downloads/ModelNet40.zip \
  --heldout-category-count 8 \
  --train-per-category 16 \
  --calibration-per-category 4 \
  --validation-per-category 4 \
  --category-ood-per-category 4 \
  --seed 1517
```

The declared manifest contains 512 train, 128 calibration, 128 validation, and
32 category-OOD records. Its expected SHA-256 is
`d44014dd2313b8815562cde9df2ba1927e1110fcfbc218428a7db39ef6b829ac`.
Generated record manifests remain ignored rather than redistributing dataset
metadata under a new policy.

## Factorial design

The full run uses at least six decoder seeds and three common refiner seeds.
Every refiner seed is crossed with every decoder seed under both baseline and
stabilized decoder arms. Arms within a decoder seed must share initialization
and the prefix of their decoder training trajectory. Refiner initialization,
data order, and optimizer budget must match across the two decoder arms.

The baseline is the raw decoder after the Experiment 017 training budget. The
stabilized candidate set consists of predeclared EMA checkpoints at epochs two,
three, and four. Candidate selection uses only K=8 FPS constellations serialized
through the real coordinate bitstream, decoded, reconstructed, and scored
against the calibration source cloud.

Candidate hashes and calibration scores must be written before validation or
OOD loaders are constructed. All decoder and refiner seeds are reported; the
experiment does not select the best model seed.

## Primary analysis

For each arm and split, first aggregate cloud loss into one log-RMSE value per
decoder/refiner cell. A balanced two-way random-effects decomposition reports
decoder, common refiner-seed, and residual interaction components. Because
there is one trained model per cell, the interaction remains descriptive.

The primary diagnosis compares decoder variance with refiner plus interaction
variance. Calibration association uses candidate-level K=8 FPS RMSE, with
Spearman association primary and Pearson secondary.

Bad-seed quality uses decoder-marginal Q90 RMSE and maximum RMSE. Q10 would be
the good tail and is not used as a failure statistic.

For the feature-codec comparison, coordinate decoder levels, coordinate refiner
levels, and the three frozen Experiment 018 feature-codec seeds are resampled
as independent training factors while category/cloud draws remain paired. Equal
numeric seed labels across different architectures do not make trained models
a scientifically paired draw.

## Gates

1. **Diagnosis:** classify decoder dominance only if the bootstrap interval for
   decoder variance divided by refiner-plus-interaction variance lies entirely
   above one; below one means refiner/interaction dominance, otherwise the
   result is inconclusive.
2. **Stability:** reduce validation decoder-marginal Q90 RMSE by at least 5%
   with a bootstrap lower bound above zero, reduce category-OOD Q90 pointwise,
   avoid worsening median absolute RMSE by more than 1%, and preserve at least
   10% median decoder-level improvement over FPS on both splits.
3. **Learned codec:** using every stabilized factorial cell and independently
   resampling all three frozen feature-codec seeds, the validation 95% interval
   for Chamfer relative RMSE improvement must exclude zero. D2 is reported
   separately and cannot substitute for this gate.
4. **Contract:** split identities are disjoint, the sealed selection record has
   no test fields, exact bitstream round trips hold, and every assigned decoder
   remains byte-identical while its refiner trains.

## Results

### Stability and matched FPS

| Split | Baseline RMSE | Stabilized RMSE | Baseline Q90 | Stabilized Q90 | Q90 reduction | Median FPS gain |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.17564 | 0.10920 | 0.19435 | 0.11130 | 42.73% | 17.26% |
| category OOD | 0.18535 | 0.12194 | 0.20976 | 0.12800 | 38.98% | 16.94% |

The worst validation decoder marginal falls from 0.20110 to 0.11172; the worst
category-OOD marginal falls from 0.21868 to 0.13075. All six decoder marginals
beat their paired baseline and matched stabilized FPS on both splits.

Fresh-resampling Q90 reductions are 42.48% on validation and 38.80% on category
OOD, so the result is not confined to the encoder-visible point sample.

### Exactly byte-matched feature codec

| Split/metric | Constellation RMSE | Feature RMSE | Improvement | 95% CI |
|---|---:|---:|---:|---:|
| validation Chamfer | 0.10920 | 0.15579 | 29.90% | 23.26 to 35.77 |
| validation fresh Chamfer | 0.10971 | 0.15574 | 29.56% | 22.76 to 35.45 |
| category-OOD Chamfer | 0.12194 | 0.17177 | 29.01% | 13.14 to 41.36 |
| category-OOD fresh Chamfer | 0.12233 | 0.17197 | 28.86% | 13.29 to 41.17 |

The bootstrap independently resamples six coordinate-decoder seeds, three
coordinate-refiner seeds, and three feature-codec seeds, while sharing
hierarchical category/cloud draws. Equal numeric seed labels across methods are
not treated as paired trained models.

### Calibration and attribution

At the one-epoch baseline, source-visible calibration FPS RMSE predicts
validation FPS quality well (Spearman 0.829, Pearson 0.971) and also tracks
category OOD (0.829 and 0.945). After stabilization, decoder quality occupies a
much narrower range and the within-range ranking is not predictive. Calibration
therefore identifies the original weak-decoder problem, but does not provide a
reliable ranking among the stabilized models.

Every decoder selected the epoch-four EMA candidate. Checkpoint selection did
not vary across seeds, so this experiment cannot attribute the improvement to
selection itself; it supports the bundled longer-training/EMA protocol.

On the fixed 16-cloud-per-split Adam diagnostic subset, stabilized recurrent
inference remains 6.3% above the source-only Adam bound on validation and 9.7%
above it on category OOD. There is useful inference headroom, but it is much
smaller than the decoder-training failure that Experiment 019 resolves.

### Contract and limitations

All 36 decoder/refiner cells completed. Decoder hashes remained unchanged,
refiner initial hashes and data-order hashes matched across arms, every message
round-tripped through the exact coordinate bitstream, every coordinate lay on
the declared lattice, every K=8 stream was 50 bytes, and Adam probes used only
source points.

Official MPEG D1/D2 were not rerun over the 36-cell factorial. The learned-codec
result above is therefore the predeclared Chamfer internal gate; one
selected official-metric pass is still required before carrying the claim into
a paper table. The feature codec is fixed-width and internal, and does not
replace comparisons with AnyPcc, UniPCGC, SparsePCGC, or another published
entropy-coded method.

## Execution

The checked-in smoke configuration validates scheduling and contracts on CPU.
The ModelNet40 configuration is intended for MPS locally or CUDA on EmpireAI.
Large checkpoints, per-cloud records, generated manifests, and downloaded data
remain ignored.

Run the full experiment and recompute statistics without retraining with:

```bash
.venv-train/bin/python -m pointconstellation.stability_experiment \
  --config configs/experiment_019_stability_modelnet40.json \
  --device mps

.venv-train/bin/python -m pointconstellation.stability_experiment \
  --config configs/experiment_019_stability_modelnet40.json \
  --aggregate-only
```

The declared local execution took 1,408.75 seconds on Apple MPS. Generated
metrics, selection records, checkpoints, per-cloud rows, and the licensed local
manifest remain ignored.

## Decision

Proceed to external published learned-codec comparisons and one selected
official D1/D2 pass. Do not invest next in a broader decoder architecture
search: longer training plus EMA already removed most of the absolute seed
instability. Preserve the six-decoder protocol as the stable Point
Constellation baseline while moving to `pcc_geo_cnn` on the aligned ModelNet
protocol and AnyPcc on a native common benchmark.
