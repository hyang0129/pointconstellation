# Experiments 017-018: ModelNet40 scale and matched learned codec

## Outcome

Experiment 017 passes the predeclared three-seed FPS gate. At the primary
`N=2048, K=8, q=12` point, the free competitive refiner improves aggregate
Chamfer RMSE over FPS through the same independently trained decoder by 16.13%
on validation (95% paired hierarchical bootstrap CI 13.49-19.85) and 18.19% on
eight held-out categories (13.37-24.59). Every seed improves and 381 of 384
validation seed-cloud pairs are wins.

Experiment 018 does **not** establish superiority over a generic learned
feature bottleneck. Against an exactly byte-matched PointNet-style feature
codec, the K=8 constellation effect is -6.97% on validation (95% CI
-29.23-10.78) and -2.83% on category OOD (-26.40-19.17). Seeds 7 and 17 beat
the feature codec, but the worse constellation seed 29 does not. Training
stability, rather than FPS allocation, is now the immediate empirical blocker.

There is a distinct surface-orientation result: official D2 RMSE favors the
constellation by 32.99% at K=8 on validation (12.03-54.99), even though
Chamfer and D1 do not. That effect remains a metric-specific observation, not
a general codec win.

## Predeclared questions

1. Does the competitive recurrent refiner beat matched-decoder FPS over three
   independent model seeds on real CAD meshes?
2. Does the low-rate result transfer to categories excluded from training?
3. Does a coordinate-only message beat a learned feature latent at exactly the
   same complete serialized size?
4. Where does the measured G-PCC frontier overtake the coordinate codec?

The primary gate for question 1 was K=8 free-coordinate Chamfer: every model
seed had to improve over FPS and the paired 95% CI had to exclude zero on both
validation and category OOD. That gate passes. The analogous exploratory gate
against the feature codec fails.

## Protocol

### Data

- source: the official Princeton
  [ModelNet40 archive](https://modelnet.cs.princeton.edu/ModelNet40.zip);
- archive SHA-256:
  `42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac`;
- generated manifest SHA-256:
  `b08bbe80713ae748e50c8b91308a14abd6240854fe1452af6027d90b35cf4a6f`;
- training: 512 official-train meshes, 16 from each of 32 categories;
- validation: 128 official-test meshes, 4 from each training category;
- category OOD: 32 official-test meshes, 4 from each of `bed`, `bottle`,
  `bowl`, `monitor`, `range_hood`, `stairs`, `stool`, and `table`;
- points: 2,048 deterministic area-weighted samples from each normalized mesh;
- fresh evaluation: a second deterministic, independent surface resampling;
- model seeds: 7, 17, and 29; fixed data seed: 1517.

Training targets are the encoder-visible source samples. Fresh target samples
are used only for evaluation. Neither the refiner nor decoder receives target
normals, fresh points, category labels, mesh identity, or hidden features.

### Coordinate codec

- free competitive recurrent refiner and frozen coordinate-only decoder;
- post-hoc unique nearest-input projection reported separately as
  `strict_subset`; it is not called a learned strict-subset encoder;
- 12-bit coordinates on the declared `[-1, 1]` lattice;
- complete 14-byte header plus fixed-width payload;
- rates K=4/8/16/32: 0.1250, 0.1953125, 0.3359375, and 0.6171875 bpp;
- 128 decoder optimizer updates followed by 128 refiner updates per seed;
- encoder-side decoder gradients see only the source cloud.

### Matched learned feature codec

Experiment 018 is an internal learned-codec control, not a published SOTA
implementation. A permutation-invariant PointNet-style encoder produces an
ordered variable-length feature prefix. An 8-bit exact quantizer and a real
12-byte-header bitstream feed a coordinate decoder. Latent dimensions
20/38/74/146 exactly match the complete stream bytes of K=4/8/16/32,
respectively. The feature codec receives 256 joint encoder/decoder updates per
seed, matching the coordinate system's total optimizer-update count.

This ablation deliberately violates the paper's coordinate-only message
contract and is labeled accordingly. It currently uses fixed-width features,
not a learned entropy model, so it is a necessary learned-representation
control but not the final external learned-codec comparison.

### Conventional codec and metrics

MPEG TMC13 v23 is run as an octree G-PCC anchor at 13 quantization points. All
stream bytes are measured from actual files. Official D1 and D2 come from the
pinned MPEG `pc_error` build. Chamfer, tail errors, Hausdorff, and sliced
Wasserstein proxies use the same reconstructed clouds and normalization.

Confidence intervals resample independent model seeds and paired cloud
identities with 10,000 hierarchical bootstrap replicates. Positive relative
improvement always means the constellation has lower aggregate RMSE than the
named baseline.

## Experiment 017: refiner versus FPS

### Free-coordinate result

| Split | K | Actual bpp | Relative RMSE improvement | 95% CI |
|---|---:|---:|---:|---:|
| validation | 4 | 0.1250 | 15.17% | 13.60 to 17.50 |
| validation | 8 | 0.1953 | 16.13% | 13.49 to 19.85 |
| validation | 16 | 0.3359 | 14.90% | 9.16 to 20.74 |
| validation | 32 | 0.6172 | 10.86% | 3.88 to 17.37 |
| category OOD | 4 | 0.1250 | 16.63% | 13.25 to 21.70 |
| category OOD | 8 | 0.1953 | 18.19% | 13.37 to 24.59 |
| category OOD | 16 | 0.3359 | 15.36% | 7.70 to 24.17 |
| category OOD | 32 | 0.6172 | 11.80% | 3.58 to 23.02 |

The post-hoc strict projection also beats FPS at every point. Its improvements
range from 9.56% to 14.34% on validation and 10.25% to 15.74% on category OOD;
all eight intervals exclude zero. This is projection headroom, not evidence
that a strict-subset encoder learned the same behavior.

The gain decreases at K=32. That agrees with the original low-rate hypothesis:
recurrent coordinate relocation helps most when only a few anchors must
communicate global structure.

### Seed sensitivity at K=8

| Seed | Validation free RMSE | Validation FPS RMSE | Category-OOD free RMSE | Category-OOD FPS RMSE |
|---:|---:|---:|---:|---:|
| 7 | 0.14454 | 0.17429 | 0.14678 | 0.18910 |
| 17 | 0.13827 | 0.17053 | 0.14237 | 0.17830 |
| 29 | 0.19356 | 0.22711 | 0.20549 | 0.24173 |

Seed 29 is substantially worse in absolute terms but still beats its matched
FPS control. This distinction matters: Experiment 017 validates the allocation
mechanism relative to the shared decoder, not reliable convergence to a strong
absolute codec.

### Per-category and representative failures

At K=8, all 32 validation categories and all 8 held-out categories improve in
the three-seed aggregate. Validation gains range from 6.55% for `keyboard` to
22.89% for `mantel`; category-OOD gains range from 11.59% for `monitor` to
21.31% for `bottle`. Every category-OOD seed-cloud pair is a win. The complete
family table is emitted as `per_family_comparisons` in the machine-readable
multi-seed artifact.

Only three of the 384 validation seed-cloud pairs regress at K=8:

| Seed | Category/model | FPS RMSE | Free RMSE |
|---:|---|---:|---:|
| 7 | `glass_box/glass_box_0266` | 0.21159 | 0.21315 |
| 17 | `night_stand/night_stand_0220` | 0.14467 | 0.14792 |
| 29 | `radio/radio_0105` | 0.28407 | 0.28604 |

These cases are selected mechanically as each seed's worst primary-point
delta, not chosen visually. The artifact records three strongest improvements
and three worst cases per split and seed under `representative_examples`, so
future qualitative rendering can use fixed identities without post-hoc
selection.

## Experiment 018: feature-latent comparison

### Chamfer/D1-like result

| Split | K-equivalent | Actual bpp | Constellation improvement | 95% CI |
|---|---:|---:|---:|---:|
| validation | 4 | 0.1250 | -23.51% | -53.33 to -0.22 |
| validation | 8 | 0.1953 | -6.97% | -29.23 to 10.78 |
| validation | 16 | 0.3359 | 5.16% | -11.63 to 19.45 |
| validation | 32 | 0.6172 | 11.98% | -1.55 to 23.10 |
| category OOD | 4 | 0.1250 | -22.15% | -49.38 to 6.00 |
| category OOD | 8 | 0.1953 | -2.83% | -26.40 to 19.17 |
| category OOD | 16 | 0.3359 | 8.61% | -13.24 to 27.72 |
| category OOD | 32 | 0.6172 | 15.17% | -5.80 to 33.67 |

The feature codec is stable across seeds but nearly rate-invariant: validation
RMSE remains around 0.146 across all four rates. The constellation improves as
K grows, but seed 29 is weak enough that no aggregate medium/high-rate interval
excludes zero. At K=8, per-seed constellation improvements are +3.26%, +9.90%,
and -30.44%.

### D2 result

Official point-to-plane distortion tells a different story on validation:

| K-equivalent | Constellation D2 RMSE improvement | 95% CI |
|---:|---:|---:|
| 4 | 29.84% | 11.73 to 48.14 |
| 8 | 32.99% | 12.03 to 54.99 |
| 16 | 33.15% | 11.22 to 54.60 |
| 32 | 28.06% | 11.62 to 43.05 |

The 32-cloud category-OOD intervals do not exclude zero. A plausible
interpretation is that decoded constellations preserve local tangent structure
better while the feature decoder minimizes pointwise set coverage. This is a
testable architectural prediction, not yet a theorem.

## G-PCC crossover

The fixed-data G-PCC curve is deterministic and is run once. On validation its
relevant measured points are:

| G-PCC bpp | Chamfer RMSE |
|---:|---:|
| 0.1891 | 0.28079 |
| 0.2139 | 0.16265 |
| 0.2394 | 0.13443 |
| 0.2729 | 0.10579 |

At K=8 and 0.1953 bpp, no sampled G-PCC point dominates any constellation
seed. Seeds 7 and 17 also strictly dominate the 0.2139-bpp G-PCC point; seed 29
does not. By K=16, G-PCC at 0.2729 bpp has lower rate and lower distortion than
all constellation seeds. The credible opportunity is therefore a narrow
low-rate regime below roughly 0.24 bpp, not the full rate curve.

No interpolation or BD-rate claim is made because the learned curve has only
four points and changes training behavior across seed. More G-PCC points around
the crossover and a stable learned model are required first.

## Runtime and shared model cost

| Method | Parameters | Checkpoint bytes | Mean encode time/cloud | Mean decode time/cloud |
|---|---:|---:|---:|---:|
| coordinate decoder + refiner | 96,038 | 405,394 | 40.44 ms | 2.32 ms |
| feature encoder + decoder | 256,085 | 1,033,091 | 2.35 ms | 2.29 ms |

The recurrent refiner is about 17 times slower to encode because it performs
two coordinated update steps with frozen-decoder gradient feedback. Shared
model bytes are reported separately and are not included in per-cloud bpp.
Both local runs stayed near 1.06 GiB peak process RSS.

## What is and is not resolved

Resolved:

- competitive recurrent allocation robustly beats FPS through the same
  decoder on external meshes, including held-out categories;
- post-hoc projection retains most of that allocation gain;
- a measurable low-rate G-PCC corridor exists on this ModelNet40 protocol;
- exact coordinate and feature streams can be compared at identical total
  bytes;
- the coordinate inductive bias has a robust validation D2 advantage over the
  internal feature decoder.

Not resolved:

- Point Constellation does not yet beat a learned feature codec reliably on
  Chamfer/D1;
- seed 29 exposes absolute decoder/refiner convergence instability;
- the feature baseline is fixed-width and internal, not entropy-coded SOTA;
- this 672-mesh subset is larger than the pilot but is not the full official
  ModelNet40 test set or an MPEG/JPEG common test condition;
- G-PCC dominates above the low-rate corridor;
- ShapeNetCore, dynamic/dense MPEG clouds, and published learned codecs remain
  external validation requirements.

## Published learned-codec replication path

The most protocol-aligned published code is
[pcc_geo_cnn](https://github.com/mauriceqch/pcc_geo_cnn), whose official
workflow explicitly trains on ModelNet point clouds and writes compressed
files. Its TensorFlow 1.13 environment requires an isolated legacy runner.

[PCGCv2](https://github.com/NJUVISION/PCGCv2) and
[SparsePCGC](https://github.com/NJUVISION/SparsePCGC) provide actual arithmetic
coding and pretrained dense-cloud models, but their official environments
require CUDA and MinkowskiEngine, and their headline testing protocols use
8i/MVUB/Owlii or LiDAR clouds rather than 2,048-point ModelNet objects. Their
results must be rerun on compatible clouds; paper table values must not be
copied into this protocol as if directly comparable.

The newer [AnyPcc](https://github.com/Wangkkklll/AnyPcc) is also a priority
external comparison because it provides full compress/decompress code and a
universal pretrained model, but it requires a CUDA/TorchSparse/DeepCABAC
environment and targets a broader dense/LiDAR benchmark suite. These external
replications belong on EmpireAI after their dataset and preprocessing contracts
are frozen.

## Reproduction

After accepting ModelNet's academic-use terms and building the pinned MPEG
tools:

```bash
bash scripts/prepare_modelnet40.sh --accept-academic-use

.venv-train/bin/python -m pointconstellation.mesh_manifest \
  --dataset modelnet40 \
  --root data/modelnet40_official/ModelNet40 \
  --output configs/manifests/modelnet40_scale.local.json \
  --archive data/downloads/ModelNet40.zip \
  --heldout-category-count 8 \
  --train-per-category 16 \
  --validation-per-category 4 \
  --category-ood-per-category 4 \
  --seed 1517

.venv-train/bin/python -m pointconstellation.standardized_multiseed \
  --config configs/experiment_017_modelnet40_multiseed.json \
  --device mps

.venv-train/bin/python -m pointconstellation.feature_codec_benchmark \
  --config configs/experiment_018_feature_codec_multiseed.json \
  --device mps
```

The local runs took 16.2 and 4.3 minutes, respectively. Downloaded archives,
extracted data, generated manifests, checkpoints, codec work directories, and
per-cloud artifacts remain ignored. Checked-in configs, source, tests, and this
report are sufficient to regenerate them.

## Decision

Continue the Experiment 005 pathway, but shift the next slice from proving an
FPS gain to improving absolute training reliability. The immediate control is
to separate decoder quality from refiner quality: train multiple decoder seeds,
select or calibrate only by training/validation-visible criteria, then train
matched refiners and re-evaluate the same fixed clouds. A result is useful only
if it raises the bad-seed floor without using held-out targets.

In parallel, prepare the isolated `pcc_geo_cnn` ModelNet replication and a
modern CUDA external-codec run. Do not scale to a larger architecture search or
claim general learned-codec superiority until one of those comparisons is
complete.
