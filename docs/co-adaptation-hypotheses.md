# Co-adaptation and constellation inference hypotheses

Last reviewed: 2026-08-11.

## Purpose

This note asks why a trained encoder can fail to emit constellations that its
decoder could reconstruct successfully. It records hypotheses before selecting
another architecture or experiment.

The working interpretation is that an encoder and decoder can settle on a
locally stable protocol. A substantially different constellation may decode
better, but reaching it can require several coordinated point movements or a
temporary increase in loss. Ordinary joint gradient descent need not discover
that transition.

That interpretation is plausible, but **co-adaptation is not yet a complete
diagnosis**. The current evidence also supports amortized-inference failure,
unstable set assignment, hard-selection gradients, and narrow useful regions
in the decoder's constellation space.

## Evidence from Experiment 004

At `N=256` and `K=16`, the frozen-decoder audit established two different
gaps:

1. The best of FPS plus eight random subsets beat the learned progressive
   selector by 34.97% on validation (`0.16820` versus `0.25866` Chamfer RMSE).
   Better messages therefore exist inside the broad input-subset
   representation class, but not necessarily inside the selector's more
   restrictive nested scalar-ranking model.
2. The separately evaluated free-coordinate condition was 15.60% below the
   best sampled-subset condition (`0.14196` versus `0.16820`). Its surface RMSE
   rose from approximately `0.00025` to `0.08374`. This difference mixes
   increased representation capacity with a possible off-manifold decoder
   exploit and is not a paired refinement gain. In the current implementation,
   "surface RMSE" is nearest-neighbor distance to the finite target sample, not
   analytic distance to the underlying continuous surface. It proves that
   coordinates moved away from observed samples, but not that they left the
   generating surface.

The first gap establishes a selector architecture, optimization, or
amortization failure; it does not yet distinguish among them. The trained model
must share one scalar ranking across all tested `N` and `K`, every smaller `K`
must be a prefix of every larger `K`, and the hard choices use a
straight-through gradient. The best-of-nine candidate is still only a sampled
feasible subset, not the optimal subset. The second gap does not by itself
prove that the encoder failed to find a valid surface constellation, because
the free-coordinate oracle has a larger hypothesis class and
continuous-surface distance was not measured.

There is also a pairing confound in the reported 15.60% free-coordinate
comparison. The `best_subset` and `free_coordinates` evaluations use different
random seeds, so the free optimizer starts from its own best-of-nine pool rather
than the exact candidate reported in the best-subset row. The aggregate values
demonstrate free-coordinate headroom, but `15.60%` is not a paired before/after
optimization gain.

The learned selector also became worse as the candidate input grew from 64 to
256 points, while FPS remained nearly stable. Combined with its poor coverage,
this implicates independent point ranking and set allocation rather than
decoder capacity alone.

## Closest results in other fields

### Amortized inference and learned compression

A feed-forward encoder replaces per-example latent optimization with one
amortized prediction. The difference between that prediction and the optimized
latent is called the *amortization gap*. [Iterative Amortized
Inference](https://proceedings.mlr.press/v80/marino18a.html) repeatedly updates
the latent using decoder-derived gradients, while [Semi-Amortized Variational
Autoencoders](https://proceedings.mlr.press/v80/kim18e.html) use an encoder to
initialize differentiable per-instance refinement.

The analogy is especially direct in [Improving Inference for Neural Image
Compression](https://proceedings.neurips.cc/paper_files/paper/2020/hash/066f182b787111ed4cb65ed437f0855b-Abstract.html).
That work identifies amortization and discretization gaps in an already trained
codec and improves rate-distortion performance by changing inference rather
than its decoder architecture.

### Auto-decoders and latent-space construction

[DeepSDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)
assigns every training shape a free latent code and optimizes those codes
jointly with the decoder. At inference time, the decoder is fixed and a new
code is optimized for the observation. This separates learning a useful code
space from learning a network that predicts codes.

For Point Constellation, the corresponding latent is not an opaque vector: it
is the quantized coordinate set itself. An auto-decoder stage could therefore
teach the decoder a deliberately occupied constellation space before an
encoder is asked to amortize inference into that space.

### Unordered set prediction and object-centric learning

[Deep Set Prediction
Networks](https://proceedings.neurips.cc/paper_files/paper/2019/hash/6e79ed05baec2754e25b4eac73a332d2-Abstract.html)
attribute prediction failures to discontinuities caused by ignoring set
structure and use an inner optimization loop to construct an output set.
[Slot
Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
instead makes exchangeable slots compete over the inputs for several rounds.
Competition creates an explaining-away mechanism: once one slot accounts for
a region, other slots are encouraged to cover something else.

[DN-DETR](https://openaccess.thecvf.com/content/CVPR2022/html/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.html)
finds that unstable bipartite assignments give set-prediction queries
inconsistent optimization targets. It stabilizes learning by perturbing known
good queries and training the model to reconstruct them.

The strict-subset branch also has close point-cloud-specific precedents.
[SampleNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.html)
predicts interacting sample proposals and differentiably projects them toward
input points. [S-NET](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dovrat_Learning_to_Sample_CVPR_2019_paper.pdf)
similarly separates freely generated simplification points from the later
matching step that produces a strict input subset.

### Discrete bottlenecks

Hard assignment can make a representation deterministic before useful roles
have been discovered. [SQ-VAE](https://proceedings.mlr.press/v162/takida22a.html)
addresses codebook collapse with stochastic quantization that gradually
self-anneals. Neural image compression has similarly used stochastic annealing
to close its discretization gap.

The direct analogue is to retain soft or stochastic point selection and point
existence early, then anneal toward the quantized coordinate set and hard
cardinality required by the codec.

### Emergent communication

An encoder, constellation, and decoder can be viewed as a speaker, message, and
listener. [Emergent Communication: Generalization and Overfitting in Lewis
Games](https://proceedings.neurips.cc/paper_files/paper/2022/hash/093b08a7ad6e6dd8d34b9cc86bb5f07c-Abstract-Conference.html)
decomposes cooperative communication into information and co-adaptation losses.
It shows that a receiver can overfit to its speaker, yielding a successful but
poorly structured protocol.

[Learning to Ground Multi-Agent Communication with
Autoencoders](https://proceedings.neurips.cc/paper/2021/hash/80fee67c8a4c4989bf8a580b4bbb0cd2-Abstract.html)
improves coordination by grounding messages in representations of the shared
environment. This supports geometric grounding, message perturbations, and
multiple decoding partners as ways to prevent an arbitrary private coordinate
language.

### Coupled optimization

Different update rates can stabilize coupled learners. The two-time-scale
update rule was developed for adversarial rather than cooperative training, so
it is an analogy rather than direct evidence, but it demonstrates that coupled
networks need not benefit from identical optimization dynamics. [GANs Trained
by a Two Time-Scale Update
Rule](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)
gives the formal result in its original setting.

## Ranked hypotheses

### H1: one-shot amortization may be insufficient

The constellation is the solution to a conditional optimization problem, not
necessarily a quantity that should be predicted correctly in one forward pass.
The encoder may provide a useful initialization while failing to perform the
multi-step search that the frozen decoder's loss surface requires.

**Prediction:** decoder-conditioned refinement reliably improves encoder
outputs without changing decoder weights. The improvement should grow with
input complexity or candidate-set size.

**Resolution class:** semi-amortized inference or a learned iterative optimizer
that consumes the input, current constellation, reconstruction residuals, and
decoder gradients.

### H2: useful decoder solutions have narrow attraction basins

Free optimization can find good coordinates, but ordinary encoder outputs may
land outside the small region from which gradients point toward them. The
off-sample oracle makes this hypothesis especially important: its gain may
come from brittle coordinate configurations rather than a reusable geometric
code. It may instead place useful points between finite target samples, which
would be legitimate; an analytic or densely resampled surface metric is needed
to distinguish those cases.

**Prediction:** small quantization-scale or Gaussian perturbations around an
oracle cause a steep loss increase, and independently optimized solutions for
nearby clouds are far apart.

**Resolution class:** train the decoder on neighborhoods of valid
constellations, use multi-scale constellation denoising, and constrain oracle
generation to the intended coordinate manifold.

### H3: independent ranking is the wrong set-allocation mechanism

Scalar importance scores do not assign distinct responsibilities to anchors.
Several high-scoring points can describe the same easy region while difficult
or small regions receive no anchor. Hard uniqueness prevents literal duplicate
indices but does not create coverage or explaining-away.

**Prediction:** selected points have overlapping receptive regions, and point
responsibilities or attention maps show little specialization. The problem
worsens as the candidate set grows, as already observed in Experiment 004.

**Resolution class:** iterative competitive slots, residual-conditioned anchor
updates, explicit point-to-anchor responsibilities, or one-to-many regional
supervision early in training.

The present progressive selector adds two specific restrictions: all rates are
prefixes of one ranking, and each point's scalar score is independent of the
points already selected. Optimal rate-distortion constellations need not be
nested. Variable-rate coding should therefore not be equated with progressive
prefix coding.

### H4: hard selection commits before a protocol is established

Top-`K`, subset selection, quantization, and existence masks introduce
discontinuities or biased surrogate gradients. An early random advantage can
become permanent, preventing anchors from trading roles or moving together.

**Prediction:** selection entropy and spatial utilization collapse early;
small score changes abruptly exchange selected points; and stochastic restarts
find materially different solutions.

**Resolution class:** stochastic or entropically regularized assignment with a
temperature schedule, followed by hardening and quantization-aware refinement.

### H5: the decoder's constellation space was not made searchable

Joint encoder/decoder training only exposes the decoder to the encoder's
current message distribution. It does not ensure that good messages form a
broad, connected, consistently interpreted space.

**Prediction:** a decoder trained jointly with per-cloud constellation
variables supports a larger fraction of good random initializations and
smoother interpolations than a decoder trained only on encoder outputs or FPS
subsets.

**Resolution class:** an auto-decoder phase that jointly learns per-cloud
constellations and the shared decoder, followed by encoder amortization and
periodic re-optimization of constellation targets.

### H6: encoder and decoder overfit a private coordinate language

Joint reconstruction rewards paired success but not interoperability,
geometric meaning, or stability outside the encoder's current distribution.
The decoder can learn shortcuts that make coordinated protocol changes costly.

**Prediction:** constellation quality falls sharply under decoder swapping,
message perturbation, or independently initialized decoders trained for the
same task.

**Resolution class:** shared geometric constraints, coordinate noise, decoder
ensembles or populations, and controlled decoder convergence. This cannot be
the sole cause of the Experiment 004 selector result because that failure
persisted with a frozen decoder.

### H7: the target constellation is non-identifiable and multimodal

An unordered cloud can have many equally useful constellations. Independently
optimized targets can permute points, exchange geometric roles, or choose
different decompositions. Directly regressing one target can average
incompatible solutions or force an arbitrary canonicalization.

**Prediction:** oracle restarts achieve similar reconstruction errors with
large matched-set distances between their constellations.

**Resolution class:** reconstruction-through-decoder training, best-of-many
hypotheses, energy-based set inference, or stochastic/diffusion refinement.
Diffusion is justified only if meaningful multimodality is measured.

### H8: joint learning uses the wrong time scale

When both networks update together, the decoder's interpretation may move
faster than the encoder can discover stable anchor roles. Freezing, alternating,
or slowing one side can reduce that moving-target effect.

**Prediction:** protocol diagnostics oscillate during joint training but
stabilize under alternating or asymmetric update schedules.

**Resolution class:** staged or alternating optimization, target decoders, or
separate learning rates. The frozen-decoder failure means this is a secondary
hypothesis rather than the leading explanation.

## Working synthesis

The highest-confidence interpretation combines H1, H2, and H3:

1. A useful constellation is found through conditional search rather than
   direct one-shot regression.
2. The decoder must make good solutions occupy wide, locally consistent
   regions rather than isolated pockets away from its training messages.
3. Constellation points must acquire distinct responsibilities through
   competition and iterative error correction.

A set transformer is a suitable implementation substrate for variable input
and output, but it does not by itself solve these problems. A diffusion-like
process is a plausible stochastic refiner when multiple distinct
constellations are genuinely valid, but it should not be introduced merely as
a larger decoder.

The long-term variable-rate codec should combine these mechanisms with point
existence or halting probabilities, a rate penalty, and the raw pass-through
endpoint. Those requirements should be layered onto a constellation inference
mechanism only after the fixed-`K` encoder can recover solutions already known
to work with a frozen decoder.

## Independent review corrections

Three independent reviews of this note and the Experiment 004 implementation
agreed on the following limits to the present evidence:

- The result identifies a selector/allocation failure, not a pure amortization
  gap or a co-adapted private protocol.
- The scalar nested ranking, progressive multi-`K` objective, softmax gradient
  dilution as `N` grows, straight-through estimator, twelve-epoch training run,
  and single seed remain confounded.
- Narrow decoder basins, multimodal oracle solutions, and decoder
  interoperability are hypotheses because no perturbation, multi-start, or
  decoder-swapping studies have measured them.
- The free-coordinate optimizer uses only sixteen Adam steps from one
  best-of-sampled initialization per evaluation stream. It is a headroom probe,
  not a converged oracle.
- Its distance metric is measured against finite target samples, and its random
  starting pool is not paired with the separately reported best-subset pool.
- The decoder copies received anchors into the reconstruction. Generated-only
  and copy-only controls are needed to determine how much of the monotonic FPS
  rate curve comes from learned completion rather than inserting more target
  points directly.
- A transformer does not automatically provide variable output. The codec must
  still make an explicit rate-distortion choice of `K` and pay to signal it.

These corrections lower confidence in H1 as a unique explanation but strengthen
the case for an experiment that separates direct per-cloud inference,
competitive allocation, and learned amortization while holding the decoder
fixed.

## Shared diagnostic preflight

The following audit should precede any of the three implementation options. It
is shared infrastructure, not a fourth architecture:

1. Fix the existing decoder, `N=256`, `K=16`, and exact 12-bit evaluation.
2. Pair every optimized run with its exact step-zero initialization. Use FPS,
   learned, random, and jittered starts over at least three seeds.
3. Record loss after `0, 1, 4, 16`, and `64` inference steps, including
   per-cloud improvement distributions rather than only aggregate RMSE.
4. Compare scalar-logit subset optimization, competitive `K x N` assignment,
   free coordinates, and surface-constrained or surface-projected coordinates.
5. Measure one-bin and multi-bin perturbation sensitivity, matched-set distance
   across restarts, selection entropy, selection turnover, coverage, minimum
   separation, and gradient norms.
6. Use the procedural parameters to evaluate distance to the analytic surface
   and reconstruct a fresh resampling of the same surface.
7. Report generated-only completion and a copy-only baseline.
8. Test whether a fixed-`N`, fixed-`K` selector can imitate FPS before training
   it through the decoder.

This preflight determines whether the primary obstruction is a poor gradient
estimator, nested-ranking misspecification, lack of competitive allocation,
ordinary amortization, or an invalid free-coordinate oracle.

## Three implementation options

### Option 1: competitive semi-amortized constellation refiner

**Risk:** medium. **Information value:** highest. **Recommendation:** implement
first if the preflight confirms robust per-cloud improvement.

Keep the Experiment 004 decoder frozen and turn the encoder into an initializer
plus a shared iterative update network:

1. Encode the variable-size input with a permutation-invariant point or set
   transformer.
2. Instantiate exactly `K` exchangeable anchor states from FPS, noisy FPS, or a
   one-shot proposal network.
3. At each of four to eight recurrent steps, construct a `K x N`
   responsibility matrix normalized so anchors compete for input regions.
4. Let anchors self-attend so their coordinate movements are coordinated.
5. Feed each update the current coordinates, input summaries, reconstruction
   residuals, and optionally the frozen decoder gradient with respect to the
   coordinates.
6. Quantize at every forward step and supervise intermediate reconstructions so
   improvement is useful even when refinement stops early.

Evaluate two output heads without changing the decoder:

- a strict-subset head using differentiable projection during training and
  unique input-point assignment at evaluation; and
- a free-coordinate head whose final quantized coordinates are transmitted
  directly.

All slot features, residuals, and gradients remain encoder-local. The bitstream
still contains only `K x 3` quantized coordinates.

**Stop/go:** require monotonic median improvement with refinement depth on
validation and parameter OOD. The strict branch should first imitate FPS within
2% reconstruction RMSE and 10% coverage RMSE. Proceed with learned refinement
only if it recovers at least half of a paired, robust direct-optimization gain
over three seeds. If direct optimization works but the learned update does not,
the remaining problem is amortization or optimizer capacity. If competitive
assignment works while scalar logits fail, H3/H4 is primary.

**Variable-`K` path:** instantiate a requested number of slots and train
non-nested constellations for sampled `K`. A later rate controller chooses `K`
by distortion plus coded rate; it need not transmit internal presence features.

### Option 2: noise-trained coordinate auto-decoder, then amortization

**Risk:** high. **Long-term upside:** high. **Recommendation:** use if the
preflight shows that the existing decoder's robust coordinate headroom is small
or its good basins are too brittle.

Construct the constellation space before asking an encoder to predict it:

1. Give each training cloud several learnable, quantized `K x 3` coordinate
   sets initialized from independent FPS, random, and jittered starts.
2. Alternate several coordinate updates with each decoder update instead of
   moving both at the same rate.
3. Train reconstruction from both clean coordinates and neighborhoods formed
   by quantization noise and multi-scale coordinate perturbations.
4. Maintain separate surface-constrained and unrestricted branches so their
   semantics and rate-distortion gains are not mixed.
5. Validate the learned space by optimizing coordinates for held-out clouds
   and fresh resamplings, not by reporting training-code reconstruction.
6. Freeze the validated decoder and train a permutation-invariant encoder or
   Option 1 refiner to amortize held-out coordinate inference.

Use oracle coordinates only as auxiliary or best-of-many supervision. Primary
training should reconstruct through the decoder so equally valid constellations
are not penalized for differing from one arbitrarily selected target set.

**Stop/go:** require at least 75% of held-out restarts to enter a predefined
good-loss region, maintain their advantage after exact quantization and
coordinate perturbations, and reconstruct fresh samples from the same
underlying geometry. The amortized encoder should recover at least half of the
held-out optimization gain. Excellent training codes with unreliable held-out
inference indicate memorization or a private decoder protocol.

**Variable-`K` path:** maintain per-cloud training constellations at several
independent `K` values with one shared decoder. Train a `K`-conditioned
amortizer and later choose among `K` values with a full rate term and raw
endpoint.

### Option 3: conditional set diffusion over robust oracle modes

**Risk and cost:** highest. **Recommendation:** conditional, not the default
next implementation.

First require multiple oracle restarts to find constellations with similar
robust distortion but large matched-set distances. Without that evidence, an
iterative deterministic refiner is the simpler explanation and implementation.

If genuine multimodality is present:

1. Build a training distribution from several quantized, validated oracle
   constellations per cloud.
2. Train a permutation-equivariant denoiser over `K` coordinate particles,
   conditioned on the input cloud, `K`, and optionally frozen-decoder gradient
   or energy features.
3. Prefer short denoising initialized from noisy FPS or a one-shot encoder over
   unconditional generation from Gaussian noise.
4. Sample several candidate constellations and choose the lowest
   rate-distortion energy under the frozen decoder.
5. If successful, distill the sampler into fewer refinement steps.

Only the final quantized coordinates are transmitted; diffusion state and
conditioning remain encoder-side computation.

**Stop/go:** do not build this option unless multi-start analysis demonstrates
several well-separated, equally valid modes. It must outperform an
equal-compute deterministic refiner and remain robust to quantization, fresh
surface resampling, and permutation. Otherwise diffusion is additional search
cost without evidence that a distributional predictor is needed.

**Variable-`K` path:** choose and signal `K` before denoising, instantiate `K`
particles, and charge the cardinality bits. Diffusion does not make cardinality
selection automatic.

## Recommended sequence

Run the shared diagnostic preflight, then implement Option 1. It is the only
path that simultaneously tests one-shot versus iterative inference and scalar
ranking versus competitive allocation without changing decoder capacity.
Option 2 is the fallback when the decoder's existing coordinate space proves
brittle or unsearchable. Option 3 is reserved for measured multimodality.

All three options now have runnable CPU prototypes. Their contracts and first
smoke results are recorded in [Experiments 005-007](experiment-005-007-prototypes.md).
The smoke results preserve this ordering: Option 1 is the next scaling target,
Option 2 is not yet competitive evidence, and Option 3 remains gated.

The subsequent five-hypothesis fan-out is also implemented. Balanced internal
transport, compression homotopy, decoder-population training, gradient-free
search, and autoregressive subset selection are documented in
[Experiments 008-012](experiment-008-012-fanout.md). The corrected scaled
refiner remains the leading path; autoregressive selection is the most useful
secondary signal because it consistently beats a matched scalar ranker, though
not FPS.

## Decision principles for the next implementation

Any candidate implementation should:

- distinguish improvements to inference from improvements to decoder capacity;
- compare one-shot encoder output, refined output, and per-cloud optimization;
- measure constellation stability under perturbation and quantization;
- preserve permutation invariance and avoid supervising an arbitrary point
  order;
- report surface distance separately from reconstruction quality;
- test multiple oracle initializations before treating one optimized target as
  ground truth; and
- delay adaptive cardinality until anchor allocation works at fixed `K`.
