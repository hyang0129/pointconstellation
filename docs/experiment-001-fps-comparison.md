# Experiment 001: matched learned constellation versus FPS

Run date: 2026-08-10

This is the first matched-rate baseline gate. It is evidence that learning the
constellation can help the shared decoder on this small procedural task; it is
not yet evidence of competitive point-cloud compression.

## Experimental control

Both runs used the same:

- 224 training clouds and 70 validation clouds, balanced across seven families;
- 256 input/output points and a 16-point constellation;
- 12 bits per coordinate, or a 576-bit / 72-byte raw payload;
- folding decoder architecture and byte-identical initial decoder weights;
- training data order, seed, quantization-jitter sequence, optimizer, and three
  epochs; and
- Apple M5 MPS device with PyTorch 2.13.0.

The only representation change was the encoder. The learned encoder regressed
surface-proximal anchors and had 119,728 parameters. The deterministic FPS
encoder selected input points and had no parameters. Both used the same 141,187
parameter decoder. Model weights are shared across clouds and are not included
in the 576-bit per-cloud payload.

Command:

```bash
.venv-train/bin/python -m pointconstellation.compare \
  --config configs/experiment_001_fps_comparison.json
```

## Result

| Validation Chamfer RMSE | Learned | FPS | Learned reduction |
|---|---:|---:|---:|
| Aggregate | 0.36926 | 0.38582 | 4.29% |
| Beam | 0.37381 | 0.38182 | 2.10% |
| Box | 0.34027 | 0.34422 | 1.15% |
| Corner | 0.28421 | 0.29421 | 3.40% |
| Cylinder | 0.35189 | 0.36286 | 3.02% |
| Pair | 0.43298 | 0.43568 | 0.62% |
| Plane | 0.33019 | 0.36544 | 9.65% |
| Sphere | 0.44483 | 0.48595 | 8.46% |

The learned constellation won in aggregate and on every family. Its anchors
also remained close to the sampled surfaces: final anchor-to-surface squared
distance was 0.01034. FPS is an input subset, so its corresponding distance was
effectively zero by construction.

## Decision

This narrowly passes the "does learning help?" sub-gate and supports continuing
the ML encoder/decoder direction. It does not pass the complete milestone gate.
The result uses one random seed, three epochs, a tiny synthetic dataset, and an
in-distribution validation split. The payload is a fixed raw coordinate count;
there is no serialized format, entropy model, metadata accounting, or external
codec comparison.

The next test should repeat the paired experiment over multiple seeds and the
`parameter_ood` split, then sweep constellation size and precision to produce a
rate-distortion curve. A learned win must survive those controls before moving
to a real dataset or making a compression claim.
