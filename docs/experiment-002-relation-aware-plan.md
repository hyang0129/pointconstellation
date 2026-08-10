# Experiment 002: relation-aware selected-rate control

Status: implemented and run locally. See the
[Experiment 002 result](experiment-002-relation-aware-result.md).

## Question

Experiment 001 showed that the learned model beat FPS at every tested rate but
did not improve when the constellation grew. This experiment tests the most
likely architectural explanation: global max pooling lets the model treat a
few coordinates as a global latent code and gives additional anchors no useful
role.

The test asks:

> Does preserving relationships between individual constellation points make
> additional coordinate rate buy additional reconstruction fidelity?

This remains a fixed-`N`, fixed-`K` experiment. Adaptive cardinality should not
be introduced until a fixed-rate model demonstrates that larger messages are
useful.

## Models

Train each model independently at `K in {4, 16, 32}` with 12-bit coordinates.

| Name | Encoder | Decoder | Purpose |
|---|---|---|---|
| `learned` | Pointwise MLP, global max pool, learned anchors | Global max pool and folding grid | Longer-run legacy control |
| `relation` | Permutation-equivariant point self-attention and `K` anchor queries | Anchor self-attention and output-query cross-attention | Proposed replacement |
| `relation_fps` | Quantized farthest-point samples | Same relation-aware decoder as `relation` | Matched-decoder sampling control |

The relation-aware decoder receives only the quantized `K x 3` coordinates.
It receives no anchor IDs, point features, family labels, or encoder state.
Neither the encoder nor decoder uses input or anchor sequence positions, so
permuting either set must not change the decoded set beyond floating-point
tolerance. Learned output queries are allowed because their order is local
decoder state and output point order is not coded.

## Data and training control

- 256 input and output points per cloud
- 12 coordinate bits, giving payloads of 144, 576, and 1,152 bits
- the same procedural training families as Experiment 001
- 448 training clouds and 140 validation clouds
- 140 clouds from the held-out `parameter_ood` split
- 12 epochs, one initial seed, and identical optimizer/loss settings across
  rates
- independently initialized models at each `K`
- one model implementation and training loop for local MPS and EmpireAI CUDA

This first run is an architectural screen, not a publication-quality estimate.
Passing it should be followed by multiple seeds and larger datasets.

## Measurements

For every model and rate, record:

- coordinate payload bits and bits per input point;
- validation and parameter-OOD Chamfer RMSE;
- validation and parameter-OOD anchor-to-surface RMSE;
- per-family distortion;
- parameter counts and training time; and
- whether the learned curve satisfies the gate below.

Inspecting anchor locations is a required follow-up if the numerical gate
passes, because off-surface coordinates can still act as an undeclared feature
channel despite quantization.

## Pass/fail gate

The `relation` curve passes only if both validation and parameter-OOD results:

1. improve from `K=4` to `K=32` by at least 1% Chamfer RMSE; and
2. have no adjacent rate step that regresses by more than 0.5%.

The report also compares `relation` with `relation_fps` at matched payloads and
with the longer-run legacy curve. Beating those controls is desirable, but the
primary gate is deliberately about rate utilization: a learned adaptive
controller has no useful cardinality decision to make if extra points do not
reduce distortion.

Failing the gate stops adaptive-`K` work on this architecture. Passing it
authorizes a multi-seed confirmation, followed by a single masked multi-`K`
model and raw-preservation mode.

## Commands

Run the complete local experiment:

```bash
python -m pointconstellation.selected_rate \
  --config configs/experiment_002_relation_aware.json
```

Resume completed model/rate points:

```bash
python -m pointconstellation.selected_rate \
  --config configs/experiment_002_relation_aware.json \
  --resume
```

Use `--device cuda` for an EmpireAI GPU allocation or `--device mps` on Apple
Silicon.
