# Experiment 030: downstream classification at matched bytes

Status: implemented; the full ModelNet40 run is pending.

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

Pending. Populate this section only from the complete full-configuration
artifact. The CPU smoke completed locally in under two minutes and its failed
gate is not a scientific result.
