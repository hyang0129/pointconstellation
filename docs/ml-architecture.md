# Core ML architecture

The analytic plane codec is not the proposed general solution. The central
experiment is a learned encoder/decoder with a geometric bottleneck:

```text
dense cloud X                 quantized constellation Z_q
N x 3        -> ML encoder -> K x 3                    -> ML decoder -> N x 3
                              no feature channels
```

## Encoder

The first implementation uses a permutation-invariant point encoder to produce
`K` anchor proposals. Proposals are softly projected toward the input surface,
quantized with a straight-through estimator, and jittered at the quantization
scale during training. All encoder features and assignment weights are then
discarded.

## Bottleneck

The tensor crossing the boundary must have shape `(batch, K, 3)`. Its rows are
shuffled during training and evaluation. No normals, features, ordering,
primitive types, topology, or per-sample model weights may cross the boundary.

## Decoder

The decoder computes any internal features it needs from the constellation
coordinates themselves. A first PointNet/FoldingNet-style decoder combines a
permutation-invariant descriptor of `Z_q` with fixed shared output queries to
generate the dense reconstruction.

The fixed queries and decoder weights are shared by the complete dataset. They
are not transmitted per cloud.

## Why begin with a simple network

The initial test isolates the coordinate-only bottleneck. Rotation augmentation
and measured rigid-transform error come first; a fully E(3)-equivariant network
is the next architecture if the simple model passes the stop/go gate. This
keeps an equivariance framework from obscuring whether constellation geometry
contains enough information at all.

The full implementation and evaluation checklist is tracked in
[Experiment 1](https://github.com/hyang0129/pointconstellation/issues/1).

## Runnable smoke configuration

`configs/experiment_001_smoke.json` exercises this architecture with 256 input
points, 16 constellation points, and 12-bit coordinate quantization. The
training command automatically selects CUDA, Apple MPS, or CPU:

```bash
python -m pointconstellation.train \
  --config configs/experiment_001_smoke.json
```

This is a pipeline and learning-signal check. The scientific experiment still
requires FPS/random baselines, all rate points, OOD splits, and actual
bitstream comparisons from Issue #1.
