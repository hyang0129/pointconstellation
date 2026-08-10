# Experiment 003: relation-decoder encoder isolation

Status: implemented and run locally. See the
[Experiment 003 result](experiment-003-encoder-isolation-result.md).

## Question

Experiment 002 showed that the relation-aware decoder uses additional
constellation points effectively, while its learned soft-projection encoder
underperformed FPS at the same rate. Experiment 003 asks:

> Can a more explicitly geometric learned selection mechanism approach the
> FPS reference without using off-surface coordinate displacement?

This is an encoder isolation at one operating point, not another rate sweep.
Use `K=16` and 12-bit coordinates, where Experiment 002 showed a large gap but
kept local iteration inexpensive.

## Controlled decoder

Every condition uses the same relation-aware decoder architecture and seed, and
the same data, optimizer, quantizer, and training budget. Each condition is
trained independently so its decoder can adapt to the encoder's coordinate
distribution. "Hold the decoder fixed" means holding its architecture,
initialization protocol, and training controls fixed; freezing an FPS-trained
decoder would confound encoder quality with a distribution mismatch.

## Encoder conditions

### FPS reference

Quantized deterministic farthest-point sampling. The encoder has no learned
parameters and provides the distortion target.

### Soft-projected relation encoder

Keep the Experiment 002 relation encoder but sweep:

- projection temperature in `{0.005, 0.02, 0.05}`; and
- anchor surface-loss weight in `{0.1, 1.0, 10.0}`.

This 3 x 3 grid separates sharper input projection from stronger loss-only
pressure. The `temperature=0.05, surface_weight=0.1` point exactly repeats the
Experiment 002 setting.

### Learned hard input-subset encoder

Add `K` learned relation queries that score the input points. Select one unique
input point per query with greedy masking. During training, use a straight-
through one-hot estimator: the forward message contains selected coordinates,
while gradients follow the masked softmax weights. Evaluation is fully hard and
deterministic.

The selected coordinates are quantized identically to FPS. No input index,
query identity, score, attention feature, or mask crosses the bottleneck.

## Configuration

- Apple MPS locally or CUDA on EmpireAI
- 256 input/output points
- `K=16`, 12 bits per coordinate, 576-bit coordinate payload
- 448 training, 140 validation, and 140 parameter-OOD clouds
- 12 epochs, batch size 8, seed 7
- one FPS reference, nine soft-projection conditions, and one hard-subset
  condition

This is a one-seed architectural screen. A passing candidate requires a later
multi-seed selected-rate confirmation.

## Measurements

Record for each condition:

- validation and parameter-OOD Chamfer RMSE;
- anchor-to-surface RMSE and repulsion;
- distortion relative to the FPS reference;
- encoder, decoder, and total parameter counts; and
- wall time and per-family metrics.

## Predeclared gate

Choose the learned candidate with the lowest validation Chamfer RMSE among
candidates whose validation anchor-to-surface RMSE is at most `0.01` in the
normalized unit sphere. The experiment passes only if that single candidate:

1. is no more than 5% worse than FPS validation Chamfer RMSE;
2. is no more than 5% worse than FPS parameter-OOD Chamfer RMSE; and
3. keeps parameter-OOD anchor-to-surface RMSE at or below `0.01`.

If no learned candidate meets the validation surface constraint, the experiment
fails without selecting a candidate. The OOD set is not used to choose among
learned candidates.

Passing authorizes a multi-seed K=4/16/32 confirmation. Failing means learned
anchor selection is not yet competitive with deterministic geometric sampling;
adaptive-`K` work remains blocked while the encoder objective or selection
mechanism is redesigned.

## Commands

```bash
.venv-train/bin/python -m pointconstellation.encoder_isolation \
  --config configs/experiment_003_encoder_isolation.json \
  --device mps
```

Add `--resume` to reuse completed condition artifacts.
