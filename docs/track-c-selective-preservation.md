# Track C: selective preservation — same-space coding of irregular geometry

Status: proposal (2026-08-27), epic #65 (sub-issues #66, #67). Third track alongside Track A (#36, codec) and
Track B (#37, representation). Extends the same-space design in
[adaptive-codec.md](adaptive-codec.md) (`X_hat = Z union G(Z)`, charged
pass-through mode, spatially adaptive mixing of preserved and generated points),
which was designed in month 1 but never executed.

## Premise

Learned codecs are lossy in a *structured* way: they regress toward the
training prior, so the parts of a cloud that deviate from that prior — defects,
thin structures, damage, unusual or anomalous geometry — are exactly the parts
that get smoothed away. Mean D1/D2/Chamfer hides this because irregular points
are a small fraction of every cloud. For downstream tasks where the anomaly *is*
the signal (defect detection, damage assessment, quality inspection), a codec
that erases irregularity destroys task value regardless of its mean PSNR.

The constellation codec emits its message in the input's own coordinate space.
That makes a preserved input point and a learned constellation point the same
kind of object: a coordinate the decoder is contracted to keep. Selective
preservation therefore needs no per-point type channel — only a mode/count
header whose bits are charged — and the error rate on the irregular subset
becomes a first-class design target rather than a casualty of averaging.

This is a representation claim, not a rate-distortion claim. Beating G-PCC is
not the objective; preserving what downstream tasks need is.

## Evidence already in hand

- Experiment 031: boundary recall halves under the constellation codec —
  irregular structure is precisely what is lost today.
- pcc_geo_cnn_v2 collapses to empty output at convergence (#15/#44): the
  extreme case of prior regression; a smoothed reconstruction is the mild case.
- Experiment 030: raw `K x 3` coordinates carry task information on their own;
  preserved raw points are the purest form of that.
- Experiment 032: decoded surfaces are stable across equivalent realizations,
  so the smooth part is well modelled and cheap; rate should go to what is not.
- Experiment 025: constellation wins below ~65 B, G-PCC above ~110 B; a
  per-region mode switch is the natural way through the crossover band, but
  that is a Track A side effect, not the gate.

## Hypothesis

H-C1. At fixed serialized bytes, a same-space codec that selectively passes
through irregular points preserves downstream anomaly-detection performance
that constellation-only and G-PCC decodes destroy, with the gain concentrated
in the high-irregularity error stratum.

Falsification: (a) if the uniform-`K` control (same bytes spent on a larger
constellation) matches the selective variant, selectivity buys nothing; (b) if
anomaly AUROC on baseline decodes is already near raw-input AUROC, the premise
that codecs erase anomalies is wrong at this rate and the track stops.

## What changes about measurement

1. **Stratified error.** D1/D2 on points binned by local irregularity
   (curvature, local density deviation, distance to the decoded smooth surface).
   The prediction is that selective preservation flattens the
   error-versus-irregularity curve at fixed bytes.
2. **Tail metrics.** 95th/99th-percentile point error and recall of
   high-irregularity points, reported alongside — never instead of — means.
3. **A task where the anomaly is the label.** 3D anomaly/defect detection with
   AUROC measured on decoded clouds versus raw input versus constellation-only
   and G-PCC decodes at matched bytes. Synthetic defects injected into
   ModelNet40 meshes first (controlled, no new data access); MVTec 3D-AD or
   Real3D-AD second if access clears.
4. **The honest control.** The same byte budget spent on a uniform larger `K`
   (and on G-PCC at matched bytes) so any win is attributable to *selective*
   preservation, not to more points.

## Fixed contract

- Preserved points are transmitted at the declared precision; the decoder
  must output them unchanged (`preservation_error = 0` by construction or
  reported if the decoder is allowed to move them).
- Mode, counts, and any selection side information are serialized and charged.
  No hidden type or feature channel per point.
- Selection sees only the source cloud. Defect labels never reach the encoder
  or decoder.
- Every arm at exactly matched bytes; validation/OOD do not select methods;
  the final slice (#25) is reserved.
- Official `pc_error` on the 12-bit grid for D1/D2; Experiment 019/020 decoder
  seeds; paired bootstrap CIs.

## Sub-experiments

1. **Experiment 040 — heuristic selective pass-through on the frozen decoder.**
   Split a byte budget between `K1` constellation points and `K2` raw points
   chosen by a source-only irregularity score (curvature, residual to the
   Chamfer decode, boundary score). Stratified and tail metrics; uniform-`K`
   and G-PCC controls. No new training. Answers falsification (a).
2. **Experiment 041 — defect-injection anomaly benchmark.** Inject synthetic
   defects (dents, holes, bumps, thin spurs) into ModelNet40 meshes with
   per-point labels; train an anomaly scorer on raw clouds; report AUROC on
   decodes from each arm at matched bytes. Answers falsification (b) and H-C1.
3. **Experiment 042 (deferred) — learned selection.** Differentiable keep
   decisions with a Lagrangian rate term, trained only if 040/041 confirm the
   premise. Not filed until Gate G-C1 passes.

## Gates

- G-C1 (after 040): on the high-irregularity stratum, selective pass-through
  reduces D1 by a CI-backed margin over the uniform-`K` control at equal bytes,
  and boundary/thin-structure recall (Experiment 031 metrics) recovers to
  ≥ 0.8 of raw-input recall. If the uniform control matches, stop.
- G-C2 (after 041): anomaly AUROC on selective decodes is within 5 points of
  raw-input AUROC and exceeds constellation-only and G-PCC decodes with CI
  excluding zero. If baseline decodes are already within 5 points of raw,
  the premise fails and the track stops.
- G-C3 (after 042, only if reached): learned selection matches or beats the
  best heuristic at equal bytes with lower encode time.

## Relation to the other tracks

Track A owns the rate curve; Track C may improve it in the 65–110 B band as a
side effect, but that is reported, not claimed. Track B owns objectives and
stability; Track C reuses its Experiment 031 surface metrics and Experiment 030
task harness. All three share the rate accounting (#21), the registry (#24),
and the final slice (#25).
