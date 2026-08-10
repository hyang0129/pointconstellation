# Experiment 001 local ML smoke result

Run date: 2026-08-10

This is a pipeline and learning-signal validation, not a compression result.

## Configuration

| Setting | Value |
|---|---:|
| Device | Apple M5 MPS |
| PyTorch | 2.13.0 |
| Input points | 256 |
| Constellation points | 16 |
| Coordinate precision | 12 bits/axis |
| Raw coordinate payload | 576 bits / 72 bytes |
| Training samples | 224 |
| Validation samples | 70 |
| Epochs | 3 |
| Shared model parameters | 260,915 |
| Wall time | 10.68 seconds |

Command:

```bash
.venv-train/bin/python -m pointconstellation.train \
  --config configs/experiment_001_smoke.json
```

## Result

| Metric | Before training | After epoch 3 |
|---|---:|---:|
| Validation loss | 0.39096 | 0.12906 |
| Validation Chamfer squared | 0.38525 | 0.12812 |
| Validation Chamfer RMSE | 0.62068 | 0.35793 |
| Anchor-to-surface loss | 0.05693 | 0.00938 |

Validation Chamfer RMSE decreased by approximately 42%. The result verifies:

- the procedural dataset supplies a usable learning signal;
- the encoder learns through surface projection and lattice quantization;
- the only encoder output passed to the decoder has shape `(B, 16, 3)`;
- the decoder reconstructs using constellation coordinates alone; and
- the complete forward/backward path runs on Apple MPS.

## What this does not establish

The smoke run has no FPS, random-sampling, feature-latent, or codec baseline. It
also reports a fixed coordinate payload rather than a complete serialized
bitstream. Consequently, it does not show that learned constellations compress
better than another representation.

The next scientific gate—learned constellation versus FPS plus the same decoder
at the same 576-bit coordinate payload—is now recorded in the
[matched FPS report](experiment-001-fps-comparison.md). Evaluation on held-out
parameter ranges remains open.
