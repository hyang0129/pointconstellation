# Experiment 015: external mesh and G-PCC pilot

## Purpose

Experiment 015 replaces the procedural generator with deterministic surface
samples from manifest-listed meshes and compares the same serialized
constellation streams with MPEG G-PCC geometry streams. It also evaluates both
with the official MPEG `pc_error` executable. The checked-in miniature meshes
exercise the complete pipeline on a MacBook; they are not a research dataset
or a state-of-the-art result.

The real-data profile targets ShapeNetCore v2. ShapeNet requires acceptance of
its terms, so the repository does not download or redistribute it. A local
manifest records model identities, split membership, relative mesh paths, and
SHA-256 hashes.

## Frozen pilot protocol

- Normalize each mesh by its bounding-box center and maximum vertex radius.
- Draw area-weighted samples from mesh triangles with piecewise face normals.
- Use `N=2048` source points and reconstruct `N=2048` points.
- Train the encoder and decoder against the exact source sample by default.
- Evaluate primary distortion against that exact source sample.
- Independently resample the same mesh for every `fresh_*` surface score.
- Never expose the independent sample, normals, category, or model identity to
  the encoder or decoder.
- Serialize constellation messages at 12 bits per coordinate for
  `K in {4, 8, 16, 32, 64, 128, 256, 512}`.
- Run TMC13 octree geometry coding at seven `codingScale` settings and count the
  compressed `.bin` bytes, including its syntax overhead.
- Run official symmetric D1 and D2, including Hausdorff variants, with
  `pc_error` on a common 12-bit integer coordinate grid.

The primary and fresh metrics answer different questions. Primary distortion
measures whether a codec reconstructed the transmitted finite point cloud.
Fresh distortion measures proximity to another finite sampling of the same
underlying mesh and reveals exact-sample overfitting. Neither is an analytic
continuous-surface distance.

This is a custom object-cloud pilot, not an MPEG common test condition. The
result therefore sets `mpeg_common_test_conditions=false` even though the
official codec and distortion software are used.

## Tool bootstrap

The bootstrap script checks out pinned official MPEG repositories and builds
them locally:

```bash
bash scripts/build_mpeg_tools.sh
```

Pinned revisions:

- TMC13: `a3d15c5e73bae20fbe2ec79be60994038a66dc8d`
- `pc_error`: `bd6df59f7a6e1176706a88b3531c9be7f5db086f`

On Apple ARM, the script applies the checked-in
`scripts/patches/dmetric-apple-arm64.patch`. It changes only platform build
flags and replaces unavailable `stat64` calls with `stat`; it does not modify
the distortion calculation. The benchmark records SHA-256 hashes of both
executables.

## Fixture result

The 2,048-point fixture run completed on Apple MPS in 25.97 seconds: 3.32
seconds for training and 22.65 seconds for bitstream-level evaluation. Peak
process RSS was 596 MiB. It used only two simple training meshes, two validation
meshes, two held-out-category meshes, and one training seed. Treat every number
below as infrastructure evidence.

The neural table shows exact-source Chamfer RMSE. Positive percentages mean the
free refiner improved over matched-decoder FPS.

| Split | K | Actual bpp | FPS | Free | Free vs FPS | Fresh free |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 4 | 0.1250 | 0.34238 | 0.33316 | +2.69% | 0.33343 |
| Validation | 8 | 0.1953 | 0.26914 | 0.26423 | +1.82% | 0.26356 |
| Validation | 16 | 0.3359 | 0.19689 | 0.19367 | +1.64% | 0.19446 |
| Validation | 32 | 0.6172 | 0.16310 | 0.16187 | +0.75% | 0.16259 |
| Validation | 64 | 1.1797 | 0.14093 | 0.14013 | +0.57% | 0.14154 |
| Validation | 128 | 2.3047 | 0.12809 | 0.12805 | +0.03% | 0.13029 |
| Validation | 256 | 4.5547 | 0.11808 | 0.11853 | -0.38% | 0.12167 |
| Validation | 512 | 9.0547 | 0.10236 | 0.10361 | -1.22% | 0.10653 |
| Category OOD | 4 | 0.1250 | 0.34669 | 0.33977 | +2.00% | 0.34277 |
| Category OOD | 8 | 0.1953 | 0.23764 | 0.23213 | +2.32% | 0.23231 |
| Category OOD | 16 | 0.3359 | 0.17881 | 0.17541 | +1.90% | 0.17265 |
| Category OOD | 32 | 0.6172 | 0.13466 | 0.13334 | +0.98% | 0.13144 |
| Category OOD | 64 | 1.1797 | 0.10636 | 0.10602 | +0.32% | 0.10447 |
| Category OOD | 128 | 2.3047 | 0.08725 | 0.08738 | -0.15% | 0.08570 |
| Category OOD | 256 | 4.5547 | 0.07226 | 0.07330 | -1.44% | 0.07254 |
| Category OOD | 512 | 9.0547 | 0.05853 | 0.06106 | -4.32% | 0.06280 |

The free refiner beats FPS only at the lower fixture rates and loses at the
largest constellations. This is consistent with Experiment 014's narrowing
low-rate advantage, but the tiny fixture cannot establish transfer.

G-PCC spans the same rate interval:

| Split | G-PCC actual bpp | Reconstructed points | Primary RMSE | Fresh RMSE | Official D1 PSNR | Official D2 PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.1777 | 6.0 | 0.46897 | 0.46856 | 16.48 | 20.70 |
| Validation | 0.2637 | 73.5 | 0.10048 | 0.10046 | 29.36 | 33.63 |
| Validation | 0.4336 | 287.5 | 0.05895 | 0.05912 | 34.42 | 38.20 |
| Validation | 1.3789 | 987.5 | 0.03069 | 0.03480 | 40.65 | 45.13 |
| Validation | 4.0820 | 1,673.0 | 0.01539 | 0.02868 | 46.96 | 51.76 |
| Validation | 7.0254 | 1,945.5 | 0.00745 | 0.02687 | 53.31 | 58.68 |
| Validation | 9.9023 | 2,020.0 | 0.00407 | 0.02673 | 58.71 | 63.21 |
| Category OOD | 0.1895 | 14.5 | 0.58344 | 0.58054 | 14.71 | 16.60 |
| Category OOD | 0.2910 | 87.5 | 0.11072 | 0.11037 | 29.02 | 31.62 |
| Category OOD | 0.5957 | 371.0 | 0.05832 | 0.05862 | 34.73 | 38.46 |
| Category OOD | 1.6895 | 1,028.5 | 0.03162 | 0.03616 | 40.52 | 43.70 |
| Category OOD | 4.5762 | 1,692.5 | 0.01493 | 0.02915 | 47.20 | 52.57 |
| Category OOD | 7.7559 | 1,949.5 | 0.00795 | 0.02817 | 52.80 | 57.30 |
| Category OOD | 10.8086 | 2,016.5 | 0.00375 | 0.02771 | 59.33 | 65.10 |

At the coarsest G-PCC point, the learned constellation has much lower
distortion at a similar rate. By the next octree point, G-PCC is already much
better. This exposes a narrow, unresolved crossover region around 0.2--0.3 bpp
that should receive denser G-PCC rate points on real data. Above that region,
G-PCC decisively dominates this shallow two-mesh model. At high G-PCC rates,
primary error approaches zero while fresh error reaches a nonzero sampling
floor, which is exactly the behavior the independent-surface metric was meant
to expose.

## Reproduction

The fixture profiles require no external data:

```bash
# Contract smoke.
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_015_mesh_fixture_smoke.json --device cpu

# Full overlapping-rate pipeline rehearsal.
.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_015_mesh_fixture_macbook.json --device mps
```

For the real pilot, place an accepted ShapeNetCore v2 copy locally and create
the untracked manifest:

```bash
.venv-train/bin/python -m pointconstellation.mesh_manifest \
  --root data/ShapeNetCore.v2 \
  --output configs/manifests/shapenetcore_v2_pilot.local.json \
  --train-categories 02691156,02958343,03001627,03636649,04379243 \
  --heldout-categories 04256520,04530566 \
  --train-per-category 8 \
  --validation-per-category 2 \
  --category-ood-per-category 5 \
  --seed 1507

.venv-train/bin/python -m pointconstellation.standardized_benchmark \
  --config configs/experiment_015_shapenetcore_pilot.json --device mps
```

Outputs include neural and G-PCC per-cloud JSONL, aggregate metrics, exact
stream rates, model/environment metadata, and executable hashes under
`artifacts/local/`.

## Remaining decision work

The ShapeNetCore run is intentionally not fabricated when licensed data is
absent. Once the local dataset is supplied, the pilot must determine whether
the low-rate FPS gain persists on held-out instances and categories, and
whether the apparent G-PCC crossover survives real geometry. Three seeds,
confidence intervals, denser rates near the crossover, a per-cloud optimization
bound, and reproducible learned-codec adapters remain necessary before a paper
claim.
