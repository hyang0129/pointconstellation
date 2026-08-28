# Month-one paper-viability sprint

## Purpose

This sprint is not the full ICLR/NeurIPS experimental program. It resolves the
uncertainties most likely to invalidate or redirect the paper before expensive
model and benchmark scaling.

The [related-work table](related-work-table.md) records the adjacent mechanism,
representation, decoder, and codec families that bound the paper claim and its
required comparisons.

The central candidate claim is that an unordered, quantized `K x 3` same-space
latent can outperform ordinary point selection because semi-amortized recurrent
inference finds decoder-useful geometric constellations. Experiment 005 supports
that claim on one procedural run. The next month must determine whether the
effect is repeatable, geometric rather than sample-specific, transferable, and
plausible at honest coded rates.

## Questions, in priority order

1. **Is the Experiment 005 gain statistically repeatable?**
2. **Does the constellation represent an underlying surface rather than exploit
   one finite target sample or decoder-specific coordinate quirks?**
3. **Does the mechanism transfer to unseen real/CAD surface distributions
   without category memorization?**
4. **Is there a plausible rate-distortion region against a standard codec once
   every transmitted bit is counted?**
5. **How much headroom remains between amortized refinement and per-cloud
   optimization?**

## Fixed arms

Use the same frozen, coordinate-only decoder wherever the comparison permits:

1. FPS plus decoder;
2. recurrent competitive refiner without decoder gradients;
3. recurrent competitive refiner with legal input-only decoder gradients;
4. final free coordinates;
5. strict unique input-subset selection or projection, labeled precisely; and
6. multi-start per-cloud Adam/STE as an inference headroom bound.

Do not add diffusion, homotopy, decoder populations, or a large architecture
sweep during this sprint.

## Work package A: corrected procedural replication

Run at least three independent decoder/refiner seeds rather than reusing one
decoder seed. Use `N in {64, 128, 256}`, `K in {4, 8, 16, 32}`, and 8-, 10-,
and 12-bit coordinates, with a predeclared primary point of `N=256, K=16,
q=12`.

Report paired per-cloud deltas, bootstrap 95% confidence intervals, median and
tail performance, convergence by refinement step, and per-primitive breakdowns.

**Gate A:** the legal gradient refiner must beat FPS on every seed at the
primary point, and the pooled paired 95% confidence interval for relative RMSE
improvement must exclude zero on validation and parameter OOD. The no-gradient
arm must improve at least one split or be explicitly dropped from the claimed
mechanism.

Status: the [Experiment 013 primary 12-bit benchmark](experiment-013-refiner-multiseed-result.md)
passes this gate across three independently trained decoder/refiner seeds. The
8- and 10-bit extensions remain open before Work Package A is complete.

## Work package B: geometry-versus-sample test

For each procedural latent surface, independently draw:

- encoder input sample `X_a`;
- training/evaluation reconstruction sample `X_b`; and
- fresh evaluation sample `X_c`.

No encoder-side computation may see `X_b` or `X_c`. Add analytic distance to
the generating surface, point-to-plane error using analytic normals, boundary
recall, thin-structure recall, and constellation perturbation tests at one and
two quantization bins.

Compare exact-sample training with independent-resampling training. Test whether
the decoder reconstructs `X_c` as well as `X_b` and whether free coordinates
remain geometrically meaningful under exact quantization.

**Gate B:** the gain over FPS must persist on fresh resampling and analytic
surface metrics. If it exists only for Chamfer against `X_b`, reframe the method
as learned sample completion/private coding rather than surface compression.

Status: [Experiment 031](experiment-031-geometry-gate.md) implements the
independent three-role sampler, analytic and feature-recall metrics, exact
lattice perturbations, and the predeclared paired gate. Its smoke validates the
execution contract. The full three-seed run remains pending, so Gate B has not
yet been declared passed or failed.

## Work package C: external surface pilot

Build a small, reproducible ShapeNetCore pilot from mesh surfaces rather than
attempting the full benchmark. Use several common training categories, held-out
instances, and at least two entirely held-out categories. Resample encoder and
target point sets independently from each mesh. Start with 2,048- or 4,096-point
clouds and spatial blocks only if memory requires them.

Train three seeds for the primary `K` values needed to establish a short rate
curve. Preserve mesh identifiers and split manifests; do not redistribute
licensed dataset files.

**Gate C:** the refiner must beat matched-decoder FPS on held-out instances and
show a nonzero gain on at least one category-held-out split. Failure only on
category-held-out data redirects the contribution toward domain-specific
compression; failure on held-out instances stops architecture scaling.

Status: ShapeNetCore execution is waiting on dataset-owner access approval.
[Experiment 016](experiment-016-modelnet40-pilot.md) pivots the same protocol to
the official ModelNet40 distribution and finds a 14--15% low-rate improvement
over FPS on all 40 pilot test meshes, including eight held-out categories. This
is encouraging external-transfer evidence, but its single model seed and tiny
per-category training subset do not complete Gate C.

[Experiment 017](experiment-017-018-modelnet40-scale.md) completes the fixed-data
three-seed ModelNet40 substitute: K=8 improves over matched-decoder FPS by
16.13% on validation (95% CI 13.49-19.85) and 18.19% on category OOD
(13.37-24.59), with every seed positive. Gate C therefore passes for the
declared ModelNet40 substitute, while the requested ShapeNetCore replication
remains pending and must not be implied by this result.

## Work package D: honest rate sanity check

Extend the existing bitstream to serialize the selected mode, bounds or
normalization metadata, `K`, precision, padding, and quantized coordinates.
Report actual bytes and bits per input point. Add an entropy-coded coordinate
variant only after the fixed-width stream is correct.

Integrate MPEG G-PCC/TMC13 and its metric software for the procedural and
ShapeNet pilot clouds. Produce at least five overlapping rate points where
possible, with D1 and D2 rate-distortion curves. This is a sanity comparison,
not yet a JPEG CTTC result.

**Gate D:** identify at least one overlapping distortion regime where the
constellation representation is plausibly competitive, or quantify the bitrate
gap that entropy modeling/variable `K` would have to close. If the gap is too
large to close even under an oracle entropy bound, stop presenting the work as
a general codec and emphasize representation or learned simplification.

Status: [Experiment 014](experiment-014-standardized-toy-benchmark.md) completes
the fixed-width stream and a MacBook-runnable 2,048-point procedural rehearsal
with six actual-rate points and bounded-memory D1/D2-style proxy metrics. It
does not complete Gate D by itself. [Experiment 015](experiment-015-external-mesh-gpcc-pilot.md)
adds manifest-backed mesh sampling, seven overlapping TMC13 rate points, and
official `pc_error` validation. The checked-in fixture exposes a possible
0.2--0.3 bpp crossover but is not scientific data. At that stage, ShapeNetCore
execution, three seeds, learned codecs, entropy coding, and a real-data rate
conclusion were all open.

[Experiments 017-018](experiment-017-018-modelnet40-scale.md) now provide three
ModelNet40 seeds, 13 measured G-PCC points, official D1/D2, and an internal
exact-byte-matched learned feature codec. They identify a plausible corridor
below about 0.24 bpp, but G-PCC dominates by K=16 and the constellation does not
reliably beat the feature codec on Chamfer.

[Experiment 019](experiment-019-stability.md) resolves the unstable absolute
learned result at the primary 50-byte point. Its six-decoder by three-refiner
factorial reduces validation bad-seed Q90 by 42.73%, retains a 17.26% median
gain over matched FPS, and beats the independently resampled three-seed feature
codec by 29.90% (95% CI 23.26--35.77). Gate D nevertheless remains open:
published entropy-coded learned codecs, a stabilized official D1/D2 pass, and
the multi-rate common-test-condition comparison are still required.

[Experiment 020](experiment-020-official-and-published-codec.md) completes the
stabilized official-metric pass at the primary rate: validation improves by
16.52% D1 and 34.65% D2, category OOD improves by 18.80% and 33.78%, and all
six decoder marginals are positive. Its pinned `pcc_geo_cnn_v2` harness is
implemented and its released five-rate one-cloud smoke is 35--176x above the
50-byte rate. A fair retrained, statistically evaluated curve with overlapping
rates remains the open part of Gate D.

## Work package E: inference headroom

At the primary operating points, run multi-start per-cloud Adam/STE from FPS and
random starts with substantially larger budgets than Experiment 011. Compare
best, median, perturbation-robust, and fresh-resampling distortion with the
amortized refiner.

**Gate E:** quantify the fraction of robust oracle improvement recovered by the
refiner. A small remaining gap favors scaling data and benchmarks; a large gap
favors better amortized inference before decoder expansion.

Status: Experiment 019's limited, fixed 16-cloud source-only Adam probe leaves
the recurrent refiner 6.3% above the bound on validation and 9.7% above it on
category OOD. This suggests that decoder stabilization was the larger immediate
failure, but it does not complete Gate E's multi-start, substantially budgeted
headroom study.

[Experiment 022](experiment-022-headroom.md) implements the full six-decoder,
four-start, 16/64/256-evaluation study with exact 50-byte streams, fresh
resampling, official D1/D2, paired uncertainty, and CPU/MPS timing. Its
predeclared Pareto gate and headline fallback are fixed; full ModelNet40
execution remains pending.

## Four-week schedule

### Week 1: freeze protocol and replicate

- implement seed manifests, paired bootstrap statistics, and primary tables;
- run Work package A;
- audit target visibility and exact quantization in tests; and
- pre-register the remaining gates in the run configuration.

### Week 2: test geometric meaning

- implement independent surface resampling and analytic metrics;
- run Work package B across the procedural families;
- add perturbation and boundary/thin-structure failure analyses; and
- decide whether free coordinates remain part of the central claim.

### Week 3: transfer pilot and rate plumbing

- implement the licensed ShapeNet mesh-sampling adapter and fixed manifests;
- launch Work package C on available GPUs;
- finish fixed-width bitstream accounting and TMC13 integration; and
- validate D1/D2 agreement with official metric software.

### Week 4: finish curves and decide

- finish ShapeNet seeds and G-PCC rate points;
- run the inference-headroom bound;
- produce tables, curves, confidence intervals, and failure cases; and
- write a decision memo choosing scale, redesign, or reframing.

## Deliverables

- immutable dataset and run manifests;
- reusable mesh-resampling and official-metric adapters;
- three-seed checkpoints and machine-readable results;
- actual bitstreams and rate-distortion curves;
- one consolidated report with all gates, including failures; and
- a concrete go/redesign/stop decision for the full paper program.

## Explicitly deferred

- full ShapeNetCore, ScanNet++, SemanticKITTI, 8iVFB, and JPEG CTTC campaigns;
- adaptive `K`, raw/pass-through mode, and learned entropy models;
- feature-latent and broad learned-codec comparisons;
- attributes, temporal coding, downstream-task preservation, and subjective
  evaluation; and
- large decoder searches, E(3)-equivariant redesigns, and diffusion.

These are paper-stage experiments only after the month-one gates justify them.

## GitHub tracking

- [#3: ICLR/NeurIPS-quality theory and benchmark epic](https://github.com/hyang0129/pointconstellation/issues/3)
- [#4: multi-seed procedural replication](https://github.com/hyang0129/pointconstellation/issues/4)
- [#5: surface representation versus sample fitting](https://github.com/hyang0129/pointconstellation/issues/5)
- [#6: ShapeNetCore external-surface pilot](https://github.com/hyang0129/pointconstellation/issues/6)
- [#7: actual bitstream and G-PCC rate sanity benchmark](https://github.com/hyang0129/pointconstellation/issues/7)
- [#8: robust per-cloud inference headroom bound](https://github.com/hyang0129/pointconstellation/issues/8)
- [#9: standardized MacBook toy benchmark](https://github.com/hyang0129/pointconstellation/issues/9)
- [#13: decoder/refiner stability decomposition](https://github.com/hyang0129/pointconstellation/issues/13)
