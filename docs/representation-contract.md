# Geometry-only representation contract

This contract makes “the information is in the geometry” testable. A result may
be labeled **geometry-only** only when it follows these rules.

## Per-cloud payload

The decoder receives one unordered multiset

```text
Z = {z_i | z_i in R^3, i = 1..K}.
```

After quantization, each `z_i` is a location on a declared 3D lattice. The
payload must not contain:

- learned or handcrafted per-point feature channels;
- normals, colors, labels, confidences, radii, or primitive IDs;
- edges, faces, parent pointers, patch assignments, or other topology;
- a meaningful point order or a choice among coordinate-equivalent orderings;
- per-cloud neural-network weights, feature grids, or implicit functions; or
- sub-quantization coordinate perturbations that disappear from the counted
  bitstream.

A global bounding box, coordinate transform, target output density, `K`, model
version, and quantizer may be codec configuration. If any varies per cloud and
is needed to decode, its coded bits count toward the rate.

## Required invariants

For a permutation `Pi` and rigid transform `T`, a conforming decoder should
satisfy:

```text
D(Pi Z) = D(Z)
D(T Z) approximately equals T D(Z)
```

The first is mandatory. The second should be an architectural property or a
measured equivariance error, not merely rotation augmentation.

## Anti-steganography controls

Coordinates are information-bearing values, so “geometry only” cannot mean
“information free.” The intended inductive bias is that constellation geometry
remains spatially meaningful. Experiments therefore need:

1. quantization in the training loop;
2. coordinate jitter at roughly half a quantization bin;
3. an anchor-to-input-surface proximity loss;
4. visualizations of constellation locations and perturbation sensitivity; and
5. performance curves as coordinate precision decreases.

If reconstruction survives only at floating-point precision or anchors visibly
leave the represented object, the model has learned a coordinate codebook, not
the proposed spatial representation.

## Rate accounting

Report at least:

- actual bitstream bytes and bits per input point;
- the quantization rule and coordinate precision;
- all headers, transforms, bounds, counts, and per-cloud metadata;
- shared decoder parameter count and binary size separately;
- encode/decode time and peak memory; and
- distortion at several rate points.

Point-count ratio and in-memory tensor size are useful diagnostics but are not
compression rates.
