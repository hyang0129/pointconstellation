# Experiment 031: surface-level geometry gate

Status: complete; Gate B passes at the Experiment 013 budget (fresh-sample Chamfer +12--16%, analytic surface RMSE +54--64%, all seeds, CIs exclude zero).

## Question and hypothesis

Experiment 031 tests whether a quantized coordinate constellation describes the
underlying procedural surface or merely helps its frozen decoder reproduce one
finite point sample. The hypothesis is that the recurrent refiner's improvement
over matched-decoder FPS persists when distortion is measured against a fresh
surface resampling and against the analytic continuous surface.

This remains a representation experiment at one diagnostic fixed rate. It is
not a general codec claim and does not replace an actual rate-distortion curve.

## Procedural protocol

Each procedural instance fixes its family parameters, normalization, and
rotation before any points are sampled. Three role-specific random streams then
draw independent point sets from that same fixed surface:

- `X_a` is the encoder input and the only point set visible to recurrent
  decoder-gradient feedback at inference time;
- `X_b` is an independent outer training/evaluation target; and
- `X_c` is a fresh evaluation-only resampling.

Every split contains plane, corner, box, sphere, capped cylinder, beam, and
separated-pair instances. Role hashes are stored in the run manifest and must be
distinct. The exact-sample training arm uses `X_a` as its outer target; the
independent-resampling arm uses `X_b`. Both arms still generate and retain all
three roles. Decoder and refiner initialization seeds, training counts, and
batch orders are matched across the two arms.

The encoder emits only the unordered quantized `K x 3` coordinate set. The
runner serializes every evaluated message through the canonical fixed-width
bitstream and checks both exact lattice membership and byte-for-byte round trip.
No target points, normals, family labels, surface parameters, or perturbation
metadata are decoder inputs.

## Metrics

Per-cloud rows report symmetric finite-sample Chamfer, point-to-plane error,
90th- and 99th-percentile point error, and Hausdorff distance against `X_a`,
`X_b`, and `X_c`. These remain sample-distance quantities.

`surface_rmse` is different: it is the one-way RMS distance from every decoded
reconstruction point to the exact finite procedural surface. The implementation
uses analytic distances to bounded plane patches, the two-patch corner, box and
beam surfaces, spheres, capped cylinders, and the separated sphere pair.
`constellation_surface_rmse` applies the same metric to the transmitted
coordinates themselves.

Boundary recall measures the fraction of fresh points in a declared boundary or
sharp-feature band that have a reconstructed point within the fixed recall
tolerance. It applies to plane, corner, box, cylinder, and beam families. Thin
structure recall applies to the beam and separated-pair families. Inapplicable
family values are `null` rather than being silently scored as perfect.

For each FPS and refiner message, deterministic tests move every coordinate by
exactly one or two quantization bins, with inward movement at lattice bounds.
Perturbed messages are reserialized and reevaluated. They are labeled
`post-hoc-lattice-perturbation`; they are not strict subsets or independently
learned free-coordinate arms.

## Gate B criterion

The predeclared Gate B comparison uses unperturbed rows. For both exact-sample
and independent-resampling training, and separately on validation and parameter
OOD:

1. the refiner must beat its matched-decoder FPS control on every model seed for
   fresh `X_c` Chamfer RMSE; and
2. the refiner must beat FPS on every model seed for analytic
   reconstruction-to-surface RMSE; and
3. the paired hierarchical 95% confidence interval for each relative RMSE
   improvement must exclude zero.

The hierarchical bootstrap pairs method values by model seed and cloud and
resamples procedural families before clouds. The checked full config declares
three model seeds. The two-seed smoke only validates execution and contracts; it
cannot decide Gate B.

Gate B passes only if all eight protocol/split/metric comparisons pass. If a
gain exists only against `X_b`, or disappears on `X_c` or analytic surface
distance, the method must be reframed as learned sample completion or a private
decoder code rather than surface compression. Boundary, thin-structure, tail,
and perturbation results are required failure analyses but are not folded into
the binary gate after observing results.

## ModelNet40 continuous-surface metrics

The mesh evaluator computes exact closest-point distance from reconstruction
points to triangles in bounded NumPy point and triangle chunks. Degenerate
faces are ignored. `surface_rmse` is the RMS of those continuous-mesh distances.
`normal_consistency` estimates an unoriented reconstruction normal by local PCA
and averages its absolute cosine with the closest non-degenerate face normal.
The unoriented definition avoids treating arbitrary mesh winding as a model
failure.

`StabilityExperimentConfig` and `OfficialStabilityConfig` expose these metrics
behind `compute_mesh_metrics`; point/triangle chunk sizes and the normal
neighborhood bound their cost. New stability and official rows also carry
per-cloud 90th percentile, 99th percentile, and Hausdorff sample errors, and
their summaries expose available tail and continuous-surface columns. Old
artifacts that predate the optional fields remain readable.

The resummary command compares Experiment 021 FPS, `random_best_of_16`, and
refiner arms with Experiment 022 source-selected multi-start Adam budgets.
Experiment 022 streams are verified and decoded directly. Experiment 021 rows
that predate stored streams are deterministically replayed from the hashed
source artifact, selector seed, and assigned frozen checkpoints. The script
writes a new artifact; it does not mutate Experiments 019--022. Comparisons use
only complete matched decoder/cloud intersections, and incomplete Experiment
022 cells are reported rather than imputed.

## Reproduction

Run the tested CPU smoke:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.geometry_gate_experiment \
  --config configs/experiment_031_geometry_gate_smoke.json \
  --device cpu
```

Run the predeclared procedural experiment only after allocating the intended
training budget:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.geometry_gate_experiment \
  --config configs/experiment_031_geometry_gate.json \
  --device mps
```

Run a bounded ModelNet40 resummary smoke against existing ignored artifacts:

```bash
/Users/hong/code/pointconstellation/.venv-train/bin/python \
  -m pointconstellation.resummarize_geometry_metrics \
  --config configs/experiment_031_geometry_resummarize.json \
  --device mps \
  --max-clouds-per-split 2
```

The full resummary omits `--max-clouds-per-split`. It requires the ignored
ModelNet40 manifest and meshes, Experiment 019 checkpoints, and completed or
partially completed Experiment 021/022 JSONL files at the configured paths.

## Results

Two runs were executed on the MacBook (MPS). A first run at the coder's light budget (140 training clouds,
2 decoder and 2 refiner epochs, width 32) failed Gate B formally on the validation surface-RMSE interval while
already showing that fresh-sample and analytic gains equal target-sample gains; it is kept under
`artifacts/local/experiment_031_geometry_gate/`. The definitive run below uses the Experiment 013 budget
(`configs/experiment_031_geometry_gate_full.json`: 448 training clouds, 12 decoder and
24 refiner epochs, width 96, 2 layers, 8 recurrent steps, model seeds
[7, 17, 29], perturbations of [1, 2, 16, 64] lattice bins, 10000 bootstrap draws; 9.6 minutes).
One code fix preceded both runs: the perturbation lattice check used a 2e-4 tolerance that float32 coordinates
cannot satisfy at 12 bits.

**Gate B passes**: for both exact-sample and independent-resampling training, the unperturbed refiner beats
matched FPS on every model seed, and the paired hierarchical 95% intervals exclude zero for fresh `X_c` Chamfer
RMSE and for analytic reconstruction-to-surface RMSE on validation and parameter OOD.

| Training protocol | Split | Metric | Refiner vs FPS | 95% CI | All seeds better |
|---|---|---|---:|---|---|
| exact_sample | validation | `x_c_chamfer_mse` | +12.82% | [6.35, 25.56] | True |
| exact_sample | validation | `surface_mse` | +59.41% | [39.07, 75.05] | True |
| exact_sample | parameter_ood | `x_c_chamfer_mse` | +15.50% | [7.47, 28.77] | True |
| exact_sample | parameter_ood | `surface_mse` | +56.35% | [43.71, 75.42] | True |
| independent_resampling | validation | `x_c_chamfer_mse` | +11.88% | [5.84, 21.88] | True |
| independent_resampling | validation | `surface_mse` | +63.56% | [44.93, 75.70] | True |
| independent_resampling | parameter_ood | `x_c_chamfer_mse` | +13.34% | [6.95, 22.82] | True |
| independent_resampling | parameter_ood | `surface_mse` | +54.23% | [34.67, 75.56] | True |

### Validation metrics under lattice perturbation

| Protocol | Method | Bins | X_b RMSE | X_c RMSE | Surface RMSE | Boundary recall | Thin recall |
|---|---|---:|---:|---:|---:|---:|---:|
| exact_sample | fps | 0 | 0.2064 | 0.2062 | 0.1185 | 0.222 | 0.291 |
| exact_sample | fps | 16 | 0.2064 | 0.2064 | 0.1187 | 0.218 | 0.293 |
| exact_sample | fps | 64 | 0.2094 | 0.2078 | 0.1178 | 0.187 | 0.263 |
| exact_sample | refiner | 0 | 0.1810 | 0.1798 | 0.0481 | 0.106 | 0.313 |
| exact_sample | refiner | 16 | 0.1809 | 0.1800 | 0.0486 | 0.105 | 0.310 |
| exact_sample | refiner | 64 | 0.1842 | 0.1825 | 0.0562 | 0.098 | 0.282 |
| independent_resampling | fps | 0 | 0.2008 | 0.2010 | 0.1105 | 0.227 | 0.298 |
| independent_resampling | fps | 16 | 0.2011 | 0.2015 | 0.1112 | 0.224 | 0.299 |
| independent_resampling | fps | 64 | 0.2014 | 0.2022 | 0.1071 | 0.199 | 0.278 |
| independent_resampling | refiner | 0 | 0.1760 | 0.1771 | 0.0403 | 0.113 | 0.316 |
| independent_resampling | refiner | 16 | 0.1764 | 0.1774 | 0.0410 | 0.112 | 0.314 |
| independent_resampling | refiner | 64 | 0.1796 | 0.1808 | 0.0500 | 0.096 | 0.279 |

### Reading

1. **The constellation represents the surface, not the sample.** Fresh-sample (`X_c`) gains equal target-sample
   (`X_b`) gains to the third decimal, training with independent resampling changes nothing, and the analytic
   distance from the reconstruction to the generating surface falls by 54--64% (0.119 -> 0.048 RMSE on
   validation). Free coordinates chosen with decoder feedback are not a decoder-private code fitted to one
   finite sample; Gate B is answered in favour of geometry.
2. **Surface gains far exceed Chamfer gains** (~60% vs ~13%): Chamfer against a finite sample under-reports how
   much closer the reconstruction sits to the true surface. This argues for reporting analytic/point-to-mesh
   distance (and D2) as primary metrics wherever a surface is available.
3. **Perturbation robustness is graceful**: 1--16-bin perturbations at 12 bits are invisible (a bin is 2.4e-4 of
   the cube), and a 64-bin perturbation (1.6% of the cube) raises surface RMSE from 0.048 to 0.056 for the
   refiner and leaves FPS essentially unchanged; there is no cliff, so the coordinates are not steganographic.
4. **Boundary recall halves for the refiner** (0.22 -> 0.11) while thin-structure recall improves slightly
   (0.29 -> 0.31): decoder-aware constellations concentrate on interior surface fidelity at the expense of
   boundary coverage, matching the light-budget run. This is a documented failure mode and a candidate
   objective term for Track B.
5. Together with Experiment 019's fresh-resampling result on meshes, the geometry-versus-sample question of the
   sprint (issue #5) is closed positively.

Artifacts: `artifacts/local/experiment_031_geometry_gate_full/geometry_gate_metrics.json`, `geometry_per_cloud.jsonl`;
light-budget run under `experiment_031_geometry_gate/`.
