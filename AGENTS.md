# Research operating instructions

## Objective

Develop Point Constellation toward an ICLR/NeurIPS-quality paper on
coordinate-only, same-space latent representations for point-cloud compression.
The intended contribution must combine a clear theoretical account of why a
quantized unordered coordinate set can communicate geometry with comprehensive,
reproducible benchmarking against strong learned, sampling, simplification, and
standard codec baselines.

The leading empirical path is the Experiment 005 competitive recurrent refiner
with a frozen decoder. Preserve competing explanations and negative results
until evidence resolves them; do not treat the current multi-seed procedural
result as a general compression claim.

## Scientific contract

- The complete per-cloud learned message is an unordered, quantized `K x 3`
  coordinate set unless an experiment is explicitly labeled as an ablation.
- Decoder inputs must not contain encoder features, target-only information,
  point order, normals, labels, primitive IDs, or per-instance weights.
- Encoder-side decoder gradients may inspect only information available to the
  encoder at inference time.
- Distinguish free-coordinate constellations, learned strict subsets, and
  post-hoc projections. Do not present one as another.
- Treat nearest distance to a finite observed cloud as a sample-distance proxy,
  not distance to the underlying continuous surface.
- Count actual serialized bits and all metadata for codec claims. Tensor size,
  `N/K`, and nominal `3Kq` alone are diagnostic rates.
- Keep shared model size separate from per-cloud rate and report amortized model
  cost where relevant.

## Paper-quality evidence

Prioritize experiments that reduce uncertainty about the central claim. Each
headline experiment should eventually include:

- predeclared hypothesis, primary metric, controls, and stop/go rule;
- at least three independent training seeds and paired per-cloud statistics;
- confidence intervals or another justified uncertainty estimate;
- ID, parameter/composition OOD, fresh-resampling, and external-dataset tests;
- actual rate-distortion curves with at least five overlapping rate points;
- D1/point-to-point and D2/point-to-plane distortion, tail error, surface and
  boundary metrics, and qualitative failure cases;
- encoder/decoder runtime, peak memory, parameter count, and model size;
- exact configurations, dataset manifests, environment capture, checkpoints,
  and machine-readable metrics sufficient to regenerate tables and figures.

Use official evaluation software and common test conditions when comparing
against MPEG G-PCC or JPEG Pleno. Compare with FPS and random sampling through
the same decoder, coordinate-only simplification methods, per-cloud optimization
bounds, feature-latent models at matched actual rate, and reproducible learned
codecs.

## Theory agenda

Develop theory alongside empirical work, without overstating what is proved.
The theory should clarify:

1. the representational capacity and ambiguities of unordered quantized
   coordinate sets;
2. invariance/equivariance requirements for encoder and decoder;
3. rate, cardinality, precision, and shared-decoder capacity tradeoffs;
4. when same-space coordinates act as geometry versus a learned private code;
5. why semi-amortized recurrent inference and decoder-gradient feedback can
   escape one-shot co-adaptation failures; and
6. conditions under which raw/pass-through coding should dominate learned
   compression.

State assumptions and distinguish formal results, testable predictions, and
empirical observations.

## Experimental workflow

Before scaling a mechanism, use the smallest decisive experiment and reuse
matched data, decoder checkpoints, initialization, optimizer budgets, and
evaluation code where scientifically valid. Fix control failures before adding
capacity. Record superseded or target-assisted results as invalid rather than
silently mixing them with corrected runs.

Prefer machine-readable run manifests and deterministic evaluation. Tests must
cover the coordinate-only contract, permutation behavior, exact quantization,
bitstream round trips, frozen-decoder integrity, and absence of target leakage.
Do not commit downloaded datasets, credentials, or large generated artifacts.

The current short-term priority is the month-one paper-viability sprint in
`docs/month-1-paper-viability-sprint.md`. Adaptive cardinality, full JPEG CTTC,
large architecture searches, temporal coding, attributes, and diffusion remain
deferred unless that sprint passes its gates.
