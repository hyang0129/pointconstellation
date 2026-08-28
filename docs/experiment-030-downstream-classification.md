# Experiment 030: downstream classification at matched bytes

Status: complete; Gate G-B1 fails (Adam-64 constellation does not beat the raw FPS subset on task utility; coordinates beat the feature latent on top-1).

## Question

Experiment 030 tests H-B1: whether the frozen per-cloud message is useful for a
downstream recognition task, rather than only for reconstruction through its
paired decoder. The task is 32-way ModelNet40 classification over the categories
used to train Experiments 018 and 019. The eight category-held-out classes are
not assigned artificial seen-class labels; they are evaluated with retrieval.

This is a representation benchmark, not evidence that downstream labels were
available to the geometry encoder. Coordinate, feature, and G-PCC encoders run
before classifier training and never receive a category, class index, model
identifier, normal, fresh surface sample, or classifier gradient.

## Data and rates

The benchmark reuses the Experiment 019 partition exactly:

- 512 official-training meshes from 32 categories for classifier training;
- 128 official-test meshes from the same 32 categories for validation; and
- 32 official-test meshes from eight disjoint categories for category OOD.

Each mesh uses the deterministic 2,048-point, area-weighted source sample from
the checked Experiment 019 protocol. The coordinate operating points are
`K={4,8,16,32}` with 12-bit coordinates. Their complete fixed-width streams,
including the 14-byte header, contain 32, 50, 86, and 158 bytes. The matched
8-bit feature streams use dimensions 20, 38, 74, and 146 and include their own
12-byte headers.

The source-cloud upper bound has no coded rate and is not placed on the finite
rate curve. G-PCC is not described as exactly byte matched: for each source and
target coordinate rate, the runner evaluates all 13 declared TMC13 points and
selects the reconstruction whose actual serialized byte count is nearest. It
reports the selected rate point and the minimum, mean, and maximum actual bytes.
There is no rate interpolation.

## Frozen representations

The checked full configuration uses Experiment 019 decoder seed 7, refiner seed
101, and Experiment 018 feature-codec seed 7. These are predeclared fixed
representation models, not classifier seeds. A full conclusion remains
conditional on this representation-model cell until other frozen model seeds
are repeated.

At every coordinate rate, the classifier receives separate inputs for:

- the raw canonical quantized FPS subset and its 2,048-point frozen-decoder
  reconstruction;
- the raw FPS-initialized, 64-evaluation Adam/STE free-coordinate constellation
  and its frozen-decoder reconstruction; and
- the raw Experiment 019 recurrent-refiner constellation and reconstruction at
  the primary `K=8` point.

The matched feature-codec latent is fed directly to its classifier. It is an
ordered feature ablation and is not labeled coordinate-only. The source cloud
is a no-rate upper bound. G-PCC contributes its decoded, variable-cardinality
point set at the nearest measured rate.

Every coordinate and feature message is serialized and decoded before it is
cached. The cache sidecar records the complete stream byte count, stream hash,
source-data identity, frozen checkpoint hashes, representation specification,
and an SHA-256 of the compressed NumPy cache. Raw constellations therefore
contain the canonical unordered-set order from the bitstream, not the encoder's
incidental point order.

## Classifiers and metrics

A small max-pooling PointNet is used for all point-set inputs, including
variable-cardinality G-PCC reconstructions. A two-layer MLP is used for feature
latents. Both use the same epoch count, batch size, optimizer settings, example
order seed, and number of optimizer updates. Parameter counts and checkpoint
hashes are reported because the appropriate PointNet and MLP do not have
identical parameterizations.

Three independent classifier seeds are trained after the representation caches
are frozen. Validation top-1 accuracy is reported with a hierarchical
classifier-seed and cloud bootstrap interval. Category-OOD top-1 is undefined:
none of its eight labels occurs in classifier training. Instead, both splits
report within-split Euclidean retrieval mAP@10 over classifier embeddings, with
the query itself excluded. Category-OOD retrieval is feasible because relevance
is defined by equality among the held-out category labels rather than by a
seen-class output head.

`downstream_metrics.json` contains per-seed predictions, correctness vectors,
per-query retrieval AP, confidence intervals, classifier histories and hashes,
actual-rate summaries, and a machine-readable rate-accuracy curve.

## Gate G-B1

The predeclared primary point is the raw 50-byte Adam-64 constellation. G-B1
passes only if that representation beats both the raw 50-byte FPS subset and
the 50-byte feature latent on both:

1. 32-way seen-category validation top-1 accuracy; and
2. category-OOD retrieval mAP@10.

Each of the four candidate-minus-baseline 95% hierarchical bootstrap intervals
must have a lower bound above zero. Decoded views, other rates, the refiner,
G-PCC, and the full source upper bound are reported controls and cannot rescue a
failed primary gate. A failure means that the tested constellation has not
shown a downstream advantage at matched bytes; it does not invalidate its
reconstruction result. A pass remains conditional on the predeclared frozen
representation-model seeds.

## Reproduction

The CPU smoke uses two meshes per category, two Adam evaluations, two classifier
epochs, and no G-PCC. It validates plumbing and contracts only; its estimates
must not be used as Experiment 030 results:

```bash
.venv-train/bin/python -m pointconstellation.downstream_classification \
  --config configs/experiment_030_downstream_classification_smoke.json \
  --device cpu
```

Run the full extraction and three-seed classification benchmark with:

```bash
.venv-train/bin/python -m pointconstellation.downstream_classification \
  --config configs/experiment_030_downstream_classification.json \
  --device mps
```

Both configurations require the ignored ModelNet40 manifests and meshes plus
the frozen Experiment 018-020 artifacts. The full G-PCC control also requires
the pinned executable configured in the JSON. Caches and classifier checkpoints
remain under `artifacts/local/experiment_030_downstream/` and are not committed.
Use `--rebuild-cache` only after intentionally changing or invalidating the
checked cache identity; a mismatched run manifest is rejected.

## Results

Executed on an EmpireAI H200 in 425 s without the optional G-PCC control (`--skip-gpcc`).
Three classifier seeds per representation; small PointNet for point sets, MLP for feature latents; the encoder
never sees labels.

**Gate G-B1 fails.** The raw Adam-64 constellation does not beat the byte-matched raw FPS subset on either
seen-category top-1 or category-OOD retrieval (both intervals include zero); it does beat the feature latent on
top-1 (+13.8 points, CI excludes zero) but not on OOD retrieval.

| Candidate | Baseline | Metric | Difference | 95% CI | Positive |
|---|---|---|---:|---|---|
| `adam64_raw_k0008` | `fps_raw_k0008` | validation_top1 | -0.021 | [-0.099, 0.057] | False |
| `adam64_raw_k0008` | `fps_raw_k0008` | ood_retrieval_map | -0.041 | [-0.143, 0.062] | False |
| `adam64_raw_k0008` | `feature_latent_k0008` | validation_top1 | +0.138 | [0.036, 0.242] | True |
| `adam64_raw_k0008` | `feature_latent_k0008` | ood_retrieval_map | +0.015 | [-0.069, 0.097] | False |

### Rate--accuracy curve (validation top-1, retrieval mAP@k)

| Representation | Bytes | Val top-1 (95% CI) | Val mAP@k | OOD mAP@k |
|---|---:|---|---:|---:|
| `adam64_decoded_k0004` | 32 | 0.328 [0.255, 0.406] | 0.216 | 0.318 |
| `adam64_raw_k0004` | 32 | 0.354 [0.279, 0.438] | 0.202 | 0.521 |
| `feature_latent_k0004` | 32 | 0.237 [0.169, 0.310] | 0.170 | 0.652 |
| `fps_decoded_k0004` | 32 | 0.201 [0.141, 0.263] | 0.146 | 0.495 |
| `fps_raw_k0004` | 32 | 0.349 [0.273, 0.430] | 0.203 | 0.570 |
| `adam64_decoded_k0008` | 50 | 0.281 [0.211, 0.354] | 0.190 | 0.523 |
| `adam64_raw_k0008` | 50 | 0.388 [0.307, 0.469] | 0.219 | 0.662 |
| `feature_latent_k0008` | 50 | 0.250 [0.177, 0.328] | 0.178 | 0.648 |
| `fps_decoded_k0008` | 50 | 0.284 [0.211, 0.362] | 0.148 | 0.451 |
| `fps_raw_k0008` | 50 | 0.409 [0.331, 0.490] | 0.232 | 0.703 |
| `refiner_decoded_k0008` | 50 | 0.258 [0.188, 0.331] | 0.143 | 0.390 |
| `refiner_raw_k0008` | 50 | 0.424 [0.346, 0.505] | 0.205 | 0.635 |
| `adam64_decoded_k0016` | 86 | 0.349 [0.268, 0.432] | 0.209 | 0.565 |
| `adam64_raw_k0016` | 86 | 0.443 [0.362, 0.524] | 0.249 | 0.699 |
| `feature_latent_k0016` | 86 | 0.247 [0.180, 0.320] | 0.181 | 0.648 |
| `fps_decoded_k0016` | 86 | 0.320 [0.242, 0.401] | 0.212 | 0.562 |
| `fps_raw_k0016` | 86 | 0.474 [0.385, 0.562] | 0.258 | 0.721 |
| `adam64_decoded_k0032` | 158 | 0.401 [0.315, 0.492] | 0.240 | 0.568 |
| `adam64_raw_k0032` | 158 | 0.451 [0.359, 0.539] | 0.266 | 0.663 |
| `feature_latent_k0032` | 158 | 0.260 [0.188, 0.336] | 0.194 | 0.659 |
| `fps_decoded_k0032` | 158 | 0.401 [0.318, 0.484] | 0.242 | 0.695 |
| `fps_raw_k0032` | 158 | 0.477 [0.388, 0.565] | 0.266 | 0.729 |
| `source_full` | full | 0.474 [0.388, 0.560] | 0.278 | 0.732 |

### Reading

1. **Raw `K x 3` coordinates are a strong task input at tens of bytes.** At 50 bytes the raw FPS subset reaches
   40.9% top-1 and the raw refiner constellation 42.4%, against 47.4% for the full 2,048-point cloud through the
   same classifier; the byte-matched feature latent reaches only 25.0%. Coordinates remain the better code for
   classification as well as for reconstruction (Experiment 023).
2. **Decoder-aware selection buys reconstruction, not task utility.** Adam-64 and FPS subsets are statistically
   indistinguishable as classifier inputs at every rate (50 B: 38.8% vs 40.9%; 86 B: 44.3% vs 47.4%), and FPS is
   marginally better on OOD retrieval. Optimizing the constellation for the frozen decoder moves points to where
   the decoder needs them, which is not where a classifier needs them.
3. **Decoded clouds are worse inputs than the raw constellation** (50 B: 28.1% decoded vs 38.8% raw for Adam;
   28.4% vs 40.9% for FPS). The shared decoder reconstructs geometry that scores well on D1/D2 but discards the
   detail a classifier uses; the K points themselves carry more discriminative information than their expansion.
4. The absolute numbers are bounded by the classifier and the 512-mesh training set (full-cloud ceiling 47.4%);
   the comparisons are paired and the conclusions are about ordering, not absolute ModelNet40 accuracy.
5. Consequence for Track B: hypothesis H-B1 holds only in its weak form (coordinates > feature latent); the
   objective-mismatch hypothesis H-B2 (Experiment 033) is now the central question, since a task-aware objective
   is what could make decoder-aware selection matter for utility.

Numbers: `artifacts/local/experiment_030_downstream/downstream_metrics.json`.
