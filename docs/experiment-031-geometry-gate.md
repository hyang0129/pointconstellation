# Experiment 031: surface-level geometry gate

Status: complete at the configured (light) budget; Gate B fails formally on the validation surface-RMSE interval, while fresh-resample and analytic gains are consistently positive and match target-sample gains.

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

Executed on the MacBook (MPS) with the configured budget: model seeds [7, 17, 29], decoder epochs
2, refiner epochs 2, 2520 per-cloud rows over the
procedural families, both training protocols (exact-sample and independent-resampling), and 0/1/2-bin lattice
perturbations. One code fix was required first: the perturbation lattice check used a 2e-4 tolerance that
float32 coordinates cannot satisfy at 12 bits; it now accepts offsets far below half a bin.

**Gate B fails under its predeclared rule** (every comparison must pass): the refiner beats FPS on fresh
`X_c` Chamfer on every split and protocol (about +3--4%, all seeds, CIs excluding zero), and on analytic
reconstruction-to-surface RMSE on parameter OOD, but the surface-RMSE interval on validation includes zero
(+2.4% [-0.07, 7.43] exact-sample; +2.1% [-0.05, 5.33] independent-resampling).

| Training protocol | Split | Metric | Refiner vs FPS | 95% CI | All seeds better | Passes |
|---|---|---|---:|---|---|---|
| exact_sample | validation | `x_c_chamfer_mse` | +3.76% | [1.71, 7.36] | True | True |
| exact_sample | validation | `surface_mse` | +2.44% | [-0.07, 7.43] | True | False |
| exact_sample | parameter_ood | `x_c_chamfer_mse` | +3.06% | [1.25, 6.17] | True | True |
| exact_sample | parameter_ood | `surface_mse` | +2.98% | [0.28, 7.11] | True | True |
| independent_resampling | validation | `x_c_chamfer_mse` | +3.63% | [1.88, 6.64] | True | True |
| independent_resampling | validation | `surface_mse` | +2.13% | [-0.05, 5.33] | True | False |
| independent_resampling | parameter_ood | `x_c_chamfer_mse` | +2.93% | [1.12, 5.52] | True | True |
| independent_resampling | parameter_ood | `surface_mse` | +2.35% | [0.10, 5.69] | True | True |

### Reading

1. **The effect is geometric, not sample fitting, but small at this budget.** Fresh-sample (`X_c`) and
   analytic surface gains track the target-sample (`X_b`) gains almost exactly (validation: 0.2753 -> 0.2657
   `X_b`, 0.2752 -> 0.2648 `X_c`, 0.2831 -> 0.2762 surface RMSE), and training with independent resampling
   changes nothing. There is no evidence of the encoder exploiting the finite target sample.
2. **The budget is far below Experiment 013's.** Absolute Chamfer RMSE is 0.26--0.28 here versus 0.145--0.19
   in Experiment 013, and the refiner gain is 3--4% versus 23%. With two decoder and two refiner epochs the
   decoder is undertrained, so this run bounds the *kind* of effect (surface-level, resampling-robust) but not
   its size. A rerun at the Experiment 013 budget is required before Gate B can be called either way.
3. **One-- and two-bin perturbations are invisible** at 12 bits (identical metrics to three decimals): a bin is
   2.4e-4 of the unit cube, so the test as configured cannot detect steganographic sensitivity. Perturbations
   of 8--64 bins, or perturbing at 8-bit precision, are the informative variants.
4. **Boundary recall drops sharply for the refiner** (0.22 -> 0.09 on validation) while thin-structure recall is
   unchanged (~0.28): decoder-aware constellations trade boundary coverage for interior fidelity. This is a
   concrete failure mode for the paper's limitations section and a candidate objective term for Track B.

Artifacts: `artifacts/local/experiment_031_geometry_gate/geometry_gate_metrics.json`, `geometry_per_cloud.jsonl`,
per-protocol subdirectories `exact_sample/` and `independent_resampling/`.

