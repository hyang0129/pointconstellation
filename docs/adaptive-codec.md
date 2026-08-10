# Adaptive same-space codec target

The fixed `N -> K -> N` autoencoder is an experimental probe, not the intended
final codec. The target is a variable-cardinality set-to-set codec:

```text
X in R^(N x 3)  -> adaptive encoder -> (mode, K, Z in R^(K x 3))
   N varies                                K varies

(mode, K, Z)    -> adaptive decoder -> X_hat in R^(M x 3)
                                             M varies
```

`Z` remains an unordered set of quantized 3D coordinates. `mode`, `N`, `K`,
precision, bounds, and entropy-coder state are codec syntax whose bits must be
counted; they are not per-point semantic feature channels.

## Compression should be optional

The encoder should minimize rate and distortion jointly, rather than force
every cloud through a predetermined bottleneck. It can choose among:

1. a small constellation for regular or redundant geometry;
2. a larger constellation for boundaries, thin structures, or irregular
   surfaces; and
3. a raw or near-raw pass-through mode when compression is not worthwhile.

The decoder should preserve every transmitted coordinate in its output, for
example `X_hat = Z union G(Z)`. In pass-through mode `G(Z)` is empty and
`Z = X`, subject only to the declared precision. This gives the codec a safe
identity endpoint instead of requiring a lossy reconstruction at every rate.

A later spatially adaptive version can mix preserved input points with learned
constellation points. That design must remain self-describing without adding a
hidden type or feature channel per point. A raw-mode flag or other syntax is
allowed only when its complete coded cost is included.

## First architecture candidate

A set/point transformer is the first practical candidate because attention and
masking naturally accept variable input and output lengths:

1. encode a variable `N` input set into contextual point features;
2. produce a ranked pool of coordinate proposals and differentiable keep
   decisions;
3. transmit only the selected, quantized coordinates plus `K`;
4. shuffle the selected set so ranking cannot become a hidden channel; and
5. decode with a variable-query set transformer or learned point-process head.

Training should sample a rate-control value and optimize a Lagrangian objective:

```text
L = distortion(X, X_hat)
  + lambda_rate * actual_or_estimated_bits(mode, headers, Z)
  + lambda_surface * anchor_surface_distance(Z, X)
  + lambda_preserve * preservation_error(Z intersection X, X_hat)
```

Learned halting, hard-concrete gates, or a monotonic ranked-token policy can
make `K` differentiable during training. The raw candidate participates in the
same mode decision, so a compressed result must beat its rate-distortion cost.

## Transformer versus diffusion

The deterministic transformer path comes first because it makes rate,
cardinality, and permutation controls easier to audit. A conditional diffusion
or flow decoder is a useful later ablation when multiple dense samplings are
valid for one constellation. It may improve surface quality and variable-density
generation, but it adds sampling cost and can obscure whether the coordinates
themselves carry the necessary structure.

Neither architecture changes the representation contract: the per-cloud
learned message is still geometry in the original coordinate space.

## Experimental ladder

1. Sweep fixed `K` and precision to measure the current rate-distortion surface.
   The [first sweep](experiment-001-rate-sweep.md) is complete and shows that
   the initial global-pooling model does not benefit from additional rate.
2. Train one masked model across several `K` values and compare it with the
   independently trained fixed-rate models.
3. Accept variable input `N` through padding masks or packed sets.
4. Add a learned `K` policy and charge its headers and entropy-coded coordinates.
5. Add variable output cardinality while preserving all transmitted points.
6. Add the raw pass-through candidate and verify that difficult inputs select
   it rather than suffering avoidable distortion.
7. Compare a deterministic transformer decoder with a conditional diffusion
   decoder at matched total rate and decode time.
