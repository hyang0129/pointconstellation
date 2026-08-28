# Draft PR integration summary

This worktree integrates the 13 requested remaining Epic 17 branches into
`agent/tracks-integration`, in the requested order. No branches were pushed and
no training jobs were run.

## Merge ledger

### 1. PR #31 — `origin/agent/epic17-selection-baselines`

Merge commit: `e9b216d`

Conflicted file:

- `src/pointconstellation/official_stability.py`: retained the selection-method
  dispatch, representation and selection metadata, and every selection result
  field while also retaining the pre-existing exact packet accounting and
  external-manifest evaluation. The experiment label fallback remains valid for
  both old and new configurations.

Validation: Ruff passed; 216 tests passed, 4 skipped.

### 2. PR #34 — `origin/agent/epic17-entropy-stream`

Merge commit: `f8a0fe1`

Conflicted file:

- `src/pointconstellation/official_stability.py`: combined membership metadata,
  selection-mode rows, and entropy-rate fields instead of choosing either row
  schema.

Behavioral interaction fixed: the entropy branch initially treated stream modes
as integers only, while PR #31 had string representation modes. The integrated
bitstream supports fixed, Rice-entropy, `free`, `strict_subset`, and `fps` modes,
keeps canonical re-encoding, and restores the representation-mode tests.

Validation: Ruff passed; 234 tests passed, 4 skipped.

### 3. PR #35 — `origin/agent/epic17-headroom-bound`

Merge commit: `9fa1bf7`

No conflicts. The headroom-bound experiment merged without a behavioral
interaction.

Validation: Ruff passed; 237 tests passed, 4 skipped.

### 4. PR #52 — `origin/agent/epic17-placement-analysis`

Merge commit: `56a7c75`

No conflicts. The placement-analysis implementation, documentation, and tests
merged cleanly.

Validation: Ruff passed; 243 tests passed, 4 skipped.

### 5. PR #54 — `origin/agent/epic17-constellation-stability`

Merge commit: `fc5ccff`

No conflicts. The constellation-stability analysis and tests merged cleanly.

Validation: Ruff passed; 247 tests passed, 4 skipped.

### 6. PR #55 — `origin/agent/epic17-objective-sweep`

Merge commit: `65de428`

No conflicts. The objective-sweep experiment, configs, documentation, and tests
merged cleanly.

Validation: Ruff passed; 258 tests passed, 4 skipped.

### 7. PR #57 — `origin/agent/epic17-rate-sweep-adam`

Merge commit: `52fc1d1`

Conflicted files:

- `src/pointconstellation/bitstream.py`: retained the entropy bit reader/writer
  and exact payload validation used by both rate paths.
- `src/pointconstellation/codecs/gpcc.py`: combined exact fixture/amortization
  accounting with the synthetic TMC13 TLV parser, leaving one canonical parser.
- `tests/test_bitstream.py`: kept all payload-length, padding, entropy, and
  round-trip coverage.
- `tests/test_gpcc.py`: kept both exact stream-breakdown fixtures and synthetic
  TLV parsing tests.

Validation: Ruff passed; 261 tests passed, 4 skipped.

### 8. PR #58 — `origin/agent/epic17-normalization-amortized`

Merge commit: `f509ede`

Conflicted files:

- `src/pointconstellation/bitstream.py`: made the binary16 center/scale suffix
  compose with fixed, representation, and Rice-entropy streams; decoding exposes
  normalized and restored original-frame coordinates and counts all 8 metadata
  bytes.
- `src/pointconstellation/feature_codec_benchmark.py`: retained decoder-file
  accounting and selection records while adding normalization-aware actual
  stream totals.
- `src/pointconstellation/official_stability.py`: combined selection and entropy
  columns with normalization and original-frame metrics.
- `src/pointconstellation/published_codec_benchmark.py`: retained diversity
  reporting and added model-amortization reporting.
- `src/pointconstellation/stability_experiment.py`: retained selection/entropy
  behavior and ensured the decoder consumes normalized coordinates while
  original-frame metrics use the serialized transform.
- `tests/test_bitstream.py`, `tests/test_official_stability.py`, and
  `tests/test_published_codec_benchmark.py`: kept both branches' tests and added
  an entropy-plus-normalization round trip.

Behavioral interaction fixed: legacy feature artifacts keep their original
`stream_bytes` field, while normalization-inclusive `total_stream_bytes` is used
for the actual-rate contract. Entropy bounds and streams include normalization
metadata when present.

Validation: Ruff passed; 271 tests passed, 4 skipped.

### 9. PR #60 — `origin/agent/epic17-bd-rate-figure`

Merge commit: `cc793e1`

Conflicted files:

- `README.md`: retained both sets of figure/table commands and explanatory
  sections.
- `src/pointconstellation/benchmark_registry.py`: combined objective, regime,
  run-seed, normalization, rate-component, model-seed, pair-ID, and RD-eligibility
  dimensions. Experiment discovery remains open-ended rather than reverting to
  a short hard-coded directory list.

Validation: Ruff passed; 276 tests passed, 5 skipped.

### 10. PR #61 — `origin/agent/epic17-external-datasets`

Merge commit: `a4416fe`

Conflicted files:

- `pyproject.toml`: retained the union of optional dependencies: the existing
  `figures` extra with `matplotlib` and the new `datasets` extra with `h5py`.
- `src/pointconstellation/data/__init__.py`: exported all pre-existing dataset
  APIs plus the external point-cloud APIs.
- `src/pointconstellation/mesh_manifest.py`: retained the ModelNet40 final-slice
  builder and added Thingi10K and ScanObjectNN manifest modes and helpers. The
  final-slice CLI still uses seed 1517 when no seed is specified.
- `tests/test_mesh_data.py`: kept final-slice, Thingi10K, and ScanObjectNN tests.

Validation: Ruff passed; 282 tests passed, 6 skipped.

### 11. PR #62 — `origin/agent/epic17-downstream-classification`

Merge commit: `4de2ccd`

No textual conflicts.

Behavioral interaction fixed: downstream tests construct synthetic
`GpccResult` values without a parsed exact breakdown, while the integrated G-PCC
codec normally provides one. `stream_breakdown` is therefore optional for
synthetic/legacy results and is validated whenever present; real `run_tmc3`
results continue to populate it.

Validation: Ruff passed; 286 tests passed, 6 skipped.

### 12. PR #63 — `origin/agent/epic17-geometry-gate`

Merge commit: `bb97342`

Conflicted files:

- `src/pointconstellation/data/__init__.py`: retained both procedural/mesh and
  raw-point-cloud dataset exports.
- `src/pointconstellation/official_stability.py`: combined selection, entropy,
  normalization, model-amortization, tail, mesh-surface, and canonical-stream
  fields. Canonical hashes are regenerated from normalized coordinates plus the
  serialized transform, so they cover the actual transmitted stream.
- `src/pointconstellation/stability_experiment.py`: retained exact packet/rate
  rows and added tail and continuous-surface metrics.

Validation: Ruff passed; 294 tests passed, 6 skipped.

### 13. PR #64 — `origin/agent/epic17-learned-entropy`

Merge commit: `b0f8237`

Conflicted files:

- `src/pointconstellation/bitstream.py`: retained fixed, Rice, string
  representation, and normalization support and added the learned arithmetic
  stream, shared model input, canonical re-encoding, CRC validation, and exact
  rate accounting.
- `src/pointconstellation/stability_experiment.py`: retained the
  normalization-inclusive Rice bound and added learned-stream bytes, bpp, model
  identity, and shared-model cost.

Behavioral interactions fixed:

- PR #64 assigned learned mode the same wire ID already used by the `free`
  representation. The public `MODE_LEARNED = 2` API is preserved, while learned
  packets use collision-free wire ID 5; wire IDs 2--4 remain the representation
  modes.
- Learned streams now encode normalized lattice coordinates, append and count
  the binary16 normalization suffix, reconstruct original-frame coordinates,
  and include normalization in the oracle bound. The learned-entropy resummary
  utility follows the same rule for external datasets.
- Added regression tests for learned/representation wire-ID separation and
  learned-stream normalization round trips.

Validation: Ruff passed; 315 tests passed, 6 skipped. The new resummary script
also passes Ruff.

## Final verification

All 13 requested remote-tracking branch tips are ancestors of `HEAD`, and the
first-parent merge history matches the requested order.

Required commands on the completed integrated source tree:

```text
/Users/hong/code/pointconstellation/.venv-train/bin/python -m ruff check src tests
All checks passed!

/Users/hong/code/pointconstellation/.venv-train/bin/python -m pytest -q -x
315 passed, 6 skipped
```

The six skips are environment-dependent optional coverage: four Matplotlib
figure/registry cases, one Draco CLI case, and one HDF5 dataset case (`h5py`).

## Subsequent draft merge

### 14. PR #69 — `origin/agent/track-exp038-regimes`

Merge commit: `abed19f`

Conflicted files:

- `src/pointconstellation/official_stability.py`: retained the integration
  branch's mesh/final-slice evaluation, normalization and original-frame
  metrics, tail/surface diagnostics, selection/entropy columns, and exact model
  and stream accounting while adding Exp 038 single-seed smoke handling,
  procedural `points`/`normals` aliases, `model_id` fallback, and legacy
  stability/official-manifest compatibility.
- `src/pointconstellation/stability_experiment.py`: retained mesh and raw-cloud
  validation, entropy and learned-stream checks, header/payload accounting, and
  decoder-model amortization while adding the one-seed non-inferential path and
  hash-checked decoder reuse. Reused decoders now also verify existing fp32/fp16
  deployment files or materialize hash-recorded deployment state dictionaries
  for legacy artifacts so the integrated model-accounting schema remains
  complete.

Behavioral interactions fixed:

- Known default fields added by Experiments 031 and 038 are normalized together
  when checking older stability artifacts; unrelated protocol differences still
  fail closed.
- Feature-codec references at a different `K`, `N`, precision, or data seed are
  recorded as `configured_reference_not_rate_matched` rather than treated as a
  comparable learned-codec result.
- Stability contract output contains both the integration branch's fixed,
  entropy, learned, and header/payload rate checks and Exp 038's decoder-reuse
  integrity check.

Validation:

- Ruff passed.
- 320 tests passed, 6 skipped for the same optional Matplotlib, Draco, and HDF5
  dependencies recorded above.
- `scripts/make_experiment_038_configs.py --check` passed for all eleven
  generated regime configs.
- The `k4_n1024` CPU smoke passed with `pc_error`: both decoder arms and both
  refiner cells completed, all stability and official contract checks passed,
  eight official rows were produced, and both inferential gates were correctly
  disabled for the single-seed smoke. Because the tracked `artifacts` symlink
  resolves outside this sandbox's writable roots, the smoke output directory
  was temporarily redirected to `/private/tmp` and the checked-in config was
  restored unchanged afterward.
