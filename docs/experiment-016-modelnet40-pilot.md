# Experiment 016: ModelNet40 external compression pilot

## Purpose

Experiment 016 executes the first real CAD-data transfer test while
ShapeNetCore access is pending. It asks whether the Experiment 005 competitive
refiner still improves over matched-decoder FPS, and where its actual
rate-distortion curve crosses MPEG G-PCC on unseen ModelNet40 meshes.

This is a one-seed stratified pilot, not the full ModelNet40 benchmark and not
an MPEG common test condition. It reduces uncertainty before a three-seed
cluster run.

## Dataset provenance and split

The archive came directly from Princeton's official ModelNet40 endpoint:

- source: `https://modelnet.cs.princeton.edu/ModelNet40.zip`;
- archive SHA-256:
  `42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac`;
- official archive counts: 9,843 training and 2,468 test OFF meshes; and
- use restriction: academic research only, with original model copyrights
  retained by their authors.

The committed preparation script requires explicit acknowledgement of the
academic-use restriction and keeps the 1.9 GiB archive and 9.1 GiB extraction
under ignored `data/`:

```bash
bash scripts/prepare_modelnet40.sh --accept-academic-use
```

The pilot manifest is deterministic but local because it contains paths and
checksums derived from the external archive. Seed 1517 partitions the 40
categories into 32 training categories and eight held-out categories:
`bed`, `bottle`, `bowl`, `monitor`, `range_hood`, `stairs`, `stool`, and
`table`.

- Training: two official-training meshes per training category, 64 total.
- Validation: one official-test mesh per training category, 32 total.
- Category OOD: one official-test mesh per held-out category, eight total.

```bash
.venv-train/bin/python -m pointconstellation.mesh_manifest \
  --dataset modelnet40 \
  --root data/modelnet40_official/ModelNet40 \
  --output configs/manifests/modelnet40_pilot.local.json \
  --archive data/downloads/ModelNet40.zip \
  --heldout-category-count 8 \
  --train-per-category 2 \
  --validation-per-category 1 \
  --category-ood-per-category 1 \
  --seed 1517
```

The resulting manifest SHA-256 is
`c5ce18c64f98f26410c4f248e31ff576dfa1de277ebdf1ca373f5d2f520f4297`.
All 104 selected meshes passed hash, parsing, normalization, and independent
surface-sampling validation. The loader explicitly handles an official archive
defect where some files concatenate the header and vertex count, such as
`OFF480`.

## Protocol

- Input and output: 2,048 points in the normalized `[-1,1]^3` domain.
- Primary target: the exact source sample available to the encoder.
- Fresh target: an independent area-weighted sample of the same mesh.
- Learned message: only a canonical, quantized, unordered `K x 3` coordinate
  set at 12 bits per axis.
- Neural rates: `K in {4,8,16,32,64,128,256,512}`.
- Controls: FPS, free input-gradient refinement, and post-hoc strict unique
  input projection through the same frozen decoder.
- G-PCC: octree geometry flags taken from the official TMC13 v23 CTC template,
  attributes disabled, and 13 measured `positionQuantizationScale` points.
  Eight settings target the sub-0.3 bpp crossover region.
- Metrics: actual serialized bpp, Chamfer/tails, fresh-surface proxies,
  official symmetric D1/D2 and Hausdorff from `pc_error`, runtime, memory, and
  per-cloud records.

## Refiner versus matched-decoder FPS

Positive values mean lower aggregate Chamfer RMSE for the free refiner.

| Split | K | Actual bpp | FPS RMSE | Free RMSE | Free improvement | Strict RMSE | Free per-cloud wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 4 | 0.1250 | 0.33152 | 0.28064 | +15.35% | 0.28659 | 32/32 |
| Validation | 8 | 0.1953 | 0.25058 | 0.21438 | +14.45% | 0.21864 | 32/32 |
| Validation | 16 | 0.3359 | 0.19166 | 0.17182 | +10.35% | 0.17401 | 32/32 |
| Validation | 32 | 0.6172 | 0.15429 | 0.14452 | +6.33% | 0.14580 | 32/32 |
| Validation | 64 | 1.1797 | 0.13090 | 0.12692 | +3.04% | 0.12745 | 31/32 |
| Validation | 128 | 2.3047 | 0.11555 | 0.11481 | +0.64% | 0.11482 | 22/32 |
| Validation | 256 | 4.5547 | 0.10376 | 0.10578 | -1.94% | 0.10457 | 4/32 |
| Validation | 512 | 9.0547 | 0.09134 | 0.09621 | -5.32% | 0.09325 | 0/32 |
| Category OOD | 4 | 0.1250 | 0.34491 | 0.29621 | +14.12% | 0.29984 | 8/8 |
| Category OOD | 8 | 0.1953 | 0.27315 | 0.23283 | +14.76% | 0.23514 | 8/8 |
| Category OOD | 16 | 0.3359 | 0.20618 | 0.18617 | +9.70% | 0.18731 | 8/8 |
| Category OOD | 32 | 0.6172 | 0.16812 | 0.15882 | +5.53% | 0.15949 | 8/8 |
| Category OOD | 64 | 1.1797 | 0.14537 | 0.14111 | +2.94% | 0.14144 | 8/8 |
| Category OOD | 128 | 2.3047 | 0.12954 | 0.12886 | +0.53% | 0.12872 | 4/8 |
| Category OOD | 256 | 4.5547 | 0.11659 | 0.11864 | -1.76% | 0.11717 | 1/8 |
| Category OOD | 512 | 9.0547 | 0.10132 | 0.10663 | -5.25% | 0.10310 | 0/8 |

The result is unusually consistent at low rate: free constellations improve all
40 unseen meshes through `K=32`, and strict projection retains most of the
gain. The benefit shrinks with `K` and reverses at `K>=256`. This localizes the
useful mechanism to aggressive compression rather than supporting a general
all-rate claim.

The first failure appears at validation `K=64` on `guitar` (-3.57%). At larger
`K`, the worst regressions concentrate on thin or highly structured objects:
`guitar`, `keyboard`, and `bench` in validation, and `stairs` in category OOD.
This suggests that the same recurrent update becomes destructive once dense
FPS already preserves fine structure; it motivates rate-conditioned update
magnitudes or an explicit no-refinement/pass-through decision rather than more
unconditional refinement steps.

Fresh-sample RMSE closely tracks primary RMSE. At validation `K=8`, free RMSE
is `0.21438` against the input and `0.21432` against an independent mesh
sample; category OOD is `0.23283` and `0.23265`. This is evidence against
finite-input memorization in this pilot, but it remains a finite-sample proxy,
not analytic continuous-surface distance.

## G-PCC crossover

The table shows the measured validation Pareto neighborhood. The
`positionQuantizationScale=1/2048` point is omitted because it is dominated by
both lower-rate and lower-distortion measurements; grid alignment makes the raw
scale sequence non-monotonic, so comparisons use measured rates and the Pareto
envelope rather than assuming scale order.

| Method/point | Actual bpp | Chamfer RMSE | Official D1 PSNR | Official D2 PSNR |
|---|---:|---:|---:|---:|
| Free constellation, K=4 | 0.1250 | 0.28064 | 19.74 | 31.47 |
| G-PCC, scale 1/1536 | 0.1875 | 0.32965 | 19.34 | 24.46 |
| G-PCC, scale 1/1280 | 0.1906 | 0.28280 | 20.67 | 25.71 |
| Free constellation, K=8 | 0.1953 | 0.21438 | 22.44 | 31.72 |
| G-PCC, scale 1/1024 | 0.2134 | 0.21014 | 23.18 | 28.67 |
| G-PCC, scale 1/768 | 0.2146 | 0.16377 | 25.30 | 30.26 |
| G-PCC, scale 1/640 | 0.2434 | 0.13581 | 26.96 | 31.88 |
| G-PCC, scale 1/512 | 0.2769 | 0.10626 | 29.06 | 34.12 |

The coordinate-only codec occupies a real but narrow low-rate niche:

- `K=4` has lower rate and lower Chamfer distortion than the first useful
  measured G-PCC point.
- At `K=8`, the nearest lower-rate G-PCC point is much worse. G-PCC at 0.2134
  bpp is about 9.2% larger and only 2.0% better in Chamfer RMSE.
- The next G-PCC point, at 0.2146 bpp, is already better than the constellation
  `K=16` point while using 36% fewer bits. Above the crossover, G-PCC dominates
  decisively.
- D2 favors the decoded smooth surface more than D1: `K=8` reaches 31.72 dB
  D2, which G-PCC does not match until roughly 0.24 bpp. This metric separation
  is scientifically interesting and should be tested with denser rates and
  more training seeds.

The same qualitative crossover appears on category OOD. At `K=8`, free
constellations achieve RMSE `0.23283` at 0.1953 bpp, while measured G-PCC moves
from `0.27202` at 0.1895 bpp to `0.22535` at 0.2212 bpp.

## Runtime and reproducibility

Run date: 2026-08-14. Device: Apple MPS. Training took 16.16 seconds and full
bitstream/official-metric evaluation took 303.79 seconds. Peak process RSS was
836.3 MiB. The decoder has 81,827 parameters and the refiner 14,211; the frozen
decoder hash remained unchanged.

```bash
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_016_modelnet40_macbook_pilot.json \
  --device mps
```

Machine-readable output is written to ignored
`artifacts/local/experiment_016_modelnet40_macbook_pilot/`, including 960
neural per-cloud rows, 520 G-PCC per-cloud rows, official metrics, executable
hashes, source/manifest identities, and environment metadata.

## Decision

The pilot passes the mechanism-transfer criterion strongly enough to justify a
three-seed ModelNet40 scale run. It does not pass issue #11's statistical gate:
there is only one model seed, only two training meshes per training category,
and one test mesh per category. No confidence interval over training seeds is
valid yet. The next run should increase training coverage, retain the fixed
category partition and source checksum, and predeclare `K=8` as the primary
low-rate point. Learned codec comparisons remain separate required work.
