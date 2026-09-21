# Benchmark expansion plan: datasets and comparable models

Status: proposal (2026-08-26). Follows the
[science review](science-review-2026-08-26.md); addresses consensus gaps 2, 3,
4, 5, and 6 (baselines, feature-latent comparator, rate curve, overlapping
published codec, evaluation set).

## Principles

1. **Stay in the declared regime.** Object-scale clouds (`N=2048`), streams of
   32--160 bytes, official `pc_error` on the common 12-bit grid, the
   Experiment 019/020 6-decoder x 3-refiner protocol. Every new dataset or
   comparator must produce at least one point inside that regime or it is a
   supplementary stress test, not a headline.
2. **One adapter per kind, not per dataset.** Meshes go through
   `data/mesh.py` + a manifest; raw point clouds need one new loader role;
   external codecs go through `codecs/external.py` (`ExternalCodecSpec`,
   black-box subprocess, hashed stream and decode). No per-baseline forks of
   the evaluation loop.
3. **Fairness rules for comparators.** Retrain on the exact training manifest
   when code allows; otherwise prove record-level disjointness from the
   released training set (the pcc_geo_cnn_v2 leakage constraint applies to
   everyone). Rate ladders must yield >=3 valid overlapping points before any
   curve is drawn; BD-rate only at >=4. Report full-stream and payload-only
   bytes. Report encode/decode time per cloud.
4. **Untouched final slice.** Nothing below is tuned on the reserved final
   test slice (see D1). It is evaluated once, at the end.

## Part A: datasets

| Tier | Dataset | Kind | Access | Role in paper | Adapter work |
|---|---|---|---|---|---|
| A1 | ModelNet40 full official test (2,468 meshes, minus the 160 used since Exp 016) | CAD mesh | on disk | **untouched final slice** for T1 | manifest only |
| A1 | ModelNet40 train, larger subset (2k--4k meshes vs 512) | CAD mesh | on disk | scale ablation: does the gain grow or shrink with decoder data | manifest only |
| A2 | **Thingi10K** (10k user-uploaded printable meshes, permissive licenses) | "in the wild" mesh | open download | held-out *distribution* OOD (not just category) | OBJ/STL -> `load_mesh` needs STL |
| A2 | **ABC** (CAD, 1M models, open) | CAD mesh | open download | second CAD source with sharp features; thin-structure failure analysis | OBJ/STEP-mesh; sample 2k models |
| A2 | ShapeNetCore v2 (issue #10) | CAD mesh | **blocked** | the reviewer-expected set; adapter and config already exist | none; run when approved |
| A3 | **ScanObjectNN** (~2.9k real scanned objects, 15 cats) | real scan, point cloud | form-gated, free | real-sensor noise/occlusion; category overlap with MN40 gives a CAD->scan transfer test | new point-cloud loader (no mesh, no analytic normals; estimate normals for D2) |
| A3 | OmniObject3D (6k real scanned objects, meshes) | real scan mesh | application | if approved, replaces ScanObjectNN as the real-object set | mesh path works |
| A4 | **8iVFB** (longdress, loot, redandblack, soldier; MPEG CTC) | dense human, ~800k pts/frame | open (MPEG) | stress test: partition each frame into 2048-point patches; report patch-level RD *and* whole-frame bytes | point-cloud loader + patch partitioner + patch-to-frame reassembly |
| A4 | Stanford scans (bunny, dragon, armadillo, happy buddha) | dense scan mesh | open | qualitative figure; single-object sanity vs G-PCC | mesh path works |
| -- | SemanticKITTI / Ford LiDAR | sparse outdoor | open | **excluded with justification**: sensor-sweep geometry, no object bottleneck, standard evaluations are 10^5 pts and >1 bpp | none |

Recommended headline set: **A1 + Thingi10K + ScanObjectNN**, with ShapeNetCore
swapped in for Thingi10K if #10 clears before the rate sweep runs. 8iVFB is a
supplementary section only.

### Dataset protocol

- Fixed manifests with SHA-256 per file, split membership, and category
  (`mesh_manifest.py`); no redistribution of licensed files.
- Independent encoder/target/fresh resampling roles per mesh (existing
  `_role_seed`). For raw point clouds the "fresh" role is a disjoint random
  subset of the scan.
- Splits: train / calibration / validation / category-OOD / **final** (once).
  OOD >=200 clouds per dataset (current 32 is why CIs cross zero).
- Decoders are trained per dataset *and* cross-evaluated (train on MN40, test
  on Thingi10K/ScanObjectNN) to separate "decoder knows the shape prior" from
  "constellation carries the geometry".

## Part B: comparable models

### B1. Non-learned selection at identical bytes (MacBook, no training)

k-means/Lloyd centroids, weighted k-means, Poisson-disk, random-start FPS,
best-of-N random subsets, multi-start Adam-STE (16/64/256 evals). All through
the frozen decoder and the same 14 B + 36 B stream. This is the cheapest and
most damaging set; run first.

### B2. Learned samplers through the frozen decoder

| Method | Why | Code | Compute |
|---|---|---|---|
| SampleNet (Lang et al. 2020) | canonical differentiable sampling; projection onto input points makes it the strict "subset" arm | PyTorch, public | ~4 H200-h, 3 seeds |
| APES (Wu et al. 2023) | attention-based, current SOTA sampler | PyTorch, public | ~4 H200-h |

### B3. Feature-latent codecs at matched bytes

| Method | Why | Code | Compute |
|---|---|---|---|
| Internal PointNet feature codec, **equal protocol** (4 ep + EMA + calibration, 6 seeds, 32/50/86/158 B) | removes the confound in Exp 019's 29.9% | in repo | ~8 H200-h |
| FoldingNet / AtlasNet AE with entropy-coded latent | external, well-known feature latent | public | ~6 H200-h |
| D-PCC (He et al. CVPR 2022) | published *point-based* learned codec with density; nearest published relative | PyTorch, public | ~10 H200-h incl. low-rate retrain |

### B4. Standard codecs

| Method | Configuration | Purpose |
|---|---|---|
| G-PCC TMC13 (have) | rerun 13 points against the *stabilized* models; add header-normalised accounting (payload vs SPS/GPS/slice bytes); sequence-amortised variant | fix the corridor artifact |
| Draco | FPS-to-K and SampleNet-to-K points, `qp` 8--12 | "learned simplification + standard coder" baseline |
| G-PCC low-rate + learned upsampler (PU-Net / Grad-PU) | G-PCC @0.24 bpp output -> upsampler | isolates the shared decoder's contribution from the refiner's |

### B5. Published learned codecs pushed to overlapping rates

Priority is feasibility at 50--160 bytes on 2048-point objects.

| Method | Family | Feasibility at our rate | Plan |
|---|---|---|---|
| pcc_geo_cnn_v2 (issue #15) | dense voxel CNN | proven: 46 B valid at 20k steps, λ=1e-5 | scale five λ to 100k steps; need >=3 valid points |
| PCGCv2 (Wang et al. 2021) | sparse conv, MinkowskiEngine | plausible at 6--7-bit voxelisation, ~0.3--1 bpp; base layer floors it | retrain low-rate on exact split; report nearest points |
| OctAttention / VoxelContext-Net | learned octree entropy | **most dangerous**: at depth 4--6 lands directly in the 0.15--0.25 bpp corridor | run at depth 4--6 on the same clouds; this is the honest G-PCC-family competitor |
| SparsePCGC | sparse, multiscale | CUDA/TorchSparse; likely floors above 0.3 bpp | supplementary |
| Pointsoup (2024) / IPDAE | point-based, lossy | closest architecture family; rates typically >1 bpp | attempt low-rate retrain; report even if no overlap |
| AnyPcc / UniPCGC (2025) | unified sparse | native benchmarks are dense/LiDAR | out of regime; cite, do not run |

Each is wired as an `ExternalCodecSpec` with its own conda env on EmpireAI
(pattern: `scripts/prepare_pcc_geo_cnn_v2.sh`), pinned commit, declared
patches with diff SHA, batch mode, independent decode, and official metrics.

## Part C: engineering work items

1. `data/pointcloud.py`: PLY/NPZ raw-cloud role with disjoint fresh subset,
   estimated normals (PCA k-NN) flagged in results as non-analytic.
2. STL loader in `data/mesh.py` (Thingi10K), mesh validity filter (watertight
   not required; drop degenerate/zero-area).
3. Patch partitioner + reassembly for 8iVFB; per-frame byte totals.
4. Header-normalised rate accounting for G-PCC (parse SPS/GPS/slice sizes) and
   for the constellation (14 B header vs payload).
5. Sorted-delta / octree entropy coder for the K constellation points
   (quantifies entropy-coding headroom; keeps fixed-width as the declared
   stream).
6. Generic torch-based `ExternalCodecSpec` template + per-codec prepare
   scripts (PCGCv2, OctAttention, D-PCC).
7. Rate-ladder search utility: bisect λ / quantisation to hit 32, 50, 86,
   158 B targets; validity = non-empty decode on all sealed clouds.
8. BD-rate/BD-PSNR with seed-bootstrap CIs, gated on >=4 overlapping points.
9. Results registry: one `benchmark_registry.jsonl` row per (dataset, split,
   method, rate point, seed) so every figure is generated from one file.
10. Final-slice runner: single invocation, refuses to run twice on the same
    manifest hash.

## Part D: phased schedule

| Phase | Weeks | Content | Compute | Gate |
|---|---|---|---|---|
| P0 | 1 | C1--C5, C7, C9; B1 on MN40 (MacBook); record #15 20k result; launch #15 100k | MacBook ~4 h; H200 50 h (#15, parallel) | refiner beats every B1 arm with CI excl. zero, else headline changes |
| P1 | 2--3 | B3 equal-protocol feature codec; B2 samplers; B4 G-PCC rerun + Draco; OctAttention depth 4--6 | H200 ~40 h | representation claim survives equal protocol; OctAttention corridor result recorded either way |
| P2 | 3--4 | Thingi10K + ScanObjectNN manifests; train 6 decoders each; cross-dataset eval; OOD >=200 | H200 ~30 h | gain over B1 arms holds on >=2 of 3 datasets |
| P3 | 4--6 | PCGCv2, D-PCC low-rate retrains; stabilized rate sweep K x q on all datasets; BD-rate where >=4 points | H200 ~60 h | >=3 overlapping points for >=2 published codecs |
| P4 | 6--7 | 8iVFB patch stress test; Stanford qualitative; ShapeNetCore if approved | H200 ~20 h | supplementary only |
| P5 | 8 | Final untouched slices, once; freeze registry; regenerate all figures | MacBook ~6 h | -- |

Total ≈ 200 H200-hours. The five live allocations (86xx namespace, 2 x 14 h and
2 x 48 h remaining at last check) cover P0--P1 immediately; P2--P4 need renewals.

## Part E: what this yields for the paper

- **T1** gains a column per dataset and a row per B1/B2 arm.
- **T2** becomes a fair equal-protocol feature-latent table with an external AE.
- **Fig 4** gets header-normalised panels, the OctAttention corridor competitor,
  and >=3 pcc_geo_cnn_v2 / PCGCv2 / D-PCC points.
- New **Fig 7**: cross-dataset transfer matrix (train dataset x eval dataset).
- New supplementary section: 8iVFB patch-level positioning, with the explicit
  statement that the method is out of the MPEG CTC regime.

## Kill criteria specific to this expansion

- OctAttention at depth 4--6 matches the constellation at 50 B: drop the RD
  corridor from the main text entirely; keep representation framing.
- Gain over k-means/Adam disappears on ScanObjectNN: the method is a CAD-prior
  effect; say so and restrict the claim to synthetic/CAD objects.
- No published codec reaches 3 valid overlapping points after P3: report the
  nearest-point gap and stop spending compute on retrains.
