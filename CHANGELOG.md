# Changelog

All notable project changes will be recorded here.

## [Unreleased]

### Added

- Geometry-only representation contract and point-cloud compression field map.
- Falsifiable experiment plan with baselines and stop/go criteria.
- Explicit ML encoder/decoder architecture for the coordinate-only experiment.
- Runnable procedural-data ML smoke experiment with a quantized `K x 3`
  bottleneck and MPS/CUDA/CPU device selection.
- Reproducible Apple MPS smoke result showing a working local learning signal.
- Deterministic quantized FPS encoder and matched-rate comparison runner.
- Per-family validation metrics and a local learned-versus-FPS result at the
  same 576-bit raw coordinate payload.
- Adaptive variable-cardinality codec target with an explicit raw pass-through
  endpoint and transformer-first architecture ladder.
- Reproducible constellation-size and coordinate-precision sweep runner.
- Local 12-point rate sweep showing learned-over-FPS gains but no learned
  fidelity improvement from increasing the current bottleneck rate.
- Experiment 002 plan, relation-aware coordinate-only transformer, selected-rate
  K=4/16/32 runner, parameter-OOD evaluation, and automated rate-curve gate.
- Local Experiment 002 result: monotonic learned curves passed the gate, longer
  training corrected the earlier flat-curve conclusion, and matched-decoder
  FPS isolated learned anchor generation as the next bottleneck.
- Four-point analytic codec for approximately planar clouds.
- NumPy Chamfer RMSE and Hausdorff metrics.
- Secret-free EmpireAI Jupyter allocation, connection, and GPU dispatch tooling.
- Synthetic wall demo, tests, linting, and continuous integration.
