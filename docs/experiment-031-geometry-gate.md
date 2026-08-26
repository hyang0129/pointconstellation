# Experiment 031: surface-level geometry gate

Status: implemented; the CPU smoke is tested. The predeclared multi-seed run and
the full ModelNet40 resummary have not been executed in this change, so Gate B
does not yet have an empirical pass/fail result.

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
