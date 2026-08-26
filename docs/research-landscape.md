# Research landscape

Last reviewed: 2026-08-10.

## Executive read

Point Constellation lies at the intersection of point-cloud compression,
surface simplification, geometric primitive coding, and point-set
autoencoders. Each area contains close ingredients, but the strict combination
appears less explored: a shared encoder/decoder whose entire per-shape latent is
an unordered, quantized set of 3D coordinates with no feature channels.

This is a novelty hypothesis, not a novelty claim. A formal literature review
and patent search would be required before publication.

The [related-work table](related-work-table.md) expands this landscape into
method-level comparisons of transmitted information, decoder type, rate regime,
and the boundary with Point Constellation.

## The field in six families

### 1. Spatial-tree and predictive geometry codecs

[MPEG G-PCC](https://www.mpeg.org/standards/MPEG-I/9/) directly codes 3D
geometry and is particularly relevant to sparse clouds. Its tools are based on
spatial decomposition and prediction. [Google
Draco](https://github.com/google/draco) is a widely used open-source codec for
meshes and point clouds and includes quantization and k-d-tree point-cloud
coding. These are essential deployment baselines.

### 2. Projection codecs

[MPEG V-PCC](https://www.mpeg.org/standards/MPEG-I/5/) projects dense 3D point
clouds into 2D patches so established video codecs can compress geometry and
attributes. It is strongest in a different operating regime from sparse LiDAR
geometry, but is a necessary dense-content comparison.

### 3. Learned occupancy and entropy models

[Learned-PCGC](https://arxiv.org/abs/1909.12037) voxelizes geometry and learns a
variational autoencoder with a hyperprior.
[VoxelContext-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Que_VoxelContext-Net_An_Octree_Based_Framework_for_Point_Cloud_Compression_CVPR_2021_paper.html)
learns entropy models over octree occupancy.
[Density-Preserving Deep Point Cloud
Compression](https://openaccess.thecvf.com/content/CVPR2022/html/He_Density-Preserving_Deep_Point_Cloud_Compression_CVPR_2022_paper.html)
downsamples points and carries learned embeddings that drive adaptive
upsampling. The published [JPEG Pleno Part
6](https://jpeg.org/jpegpleno/index.html) standard confirms that learned point
cloud coding is now part of the standards landscape.

These methods optimize real rate-distortion objectives. Point Constellation
must eventually use the same standard of evidence.

### 4. Implicit neural representations

Implicit codecs represent a surface or occupancy field with a latent code or
per-instance network. The 2023 [Implicit
Autoencoder](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Implicit_Autoencoder_for_Point-Cloud_Self-Supervised_Representation_Learning_ICCV_2023_paper.html)
argues that reconstructing the underlying continuous surface avoids learning
the accidental sampling pattern of an input cloud. [LINR-PCGC
(2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_LINR-PCGC_Lossless_Implicit_Neural_Representations_for_Point_Cloud_Geometry_Compression_ICCV_2025_paper.html)
encodes overfit implicit-network parameters for lossless geometry coding.

This family supports the premise that the target should be underlying geometry,
not exact resampling, but its transmitted representation is not a pure 3D point
set.

### 5. Simplification and learned sampling

[SampleNet](https://arxiv.org/abs/1912.03663) learns differentiable,
task-specific sampling and evaluates geometry reconstruction. [Learnable
Feature-Preserving
Simplification](https://arxiv.org/abs/2109.14982) selects and repositions a
user-defined number of points while minimizing perceptual error.
[PCS-Net](https://arxiv.org/abs/2203.09088) learns simplification for subsequent
surface reconstruction.

These methods are probably the strongest coordinate-only ablations. Their
typical goal is to preserve enough samples for an existing task or surface
reconstructor, rather than to learn a strict geometry-only compression
bottleneck and coded bitstream.

### 6. Primitive and surface coding

The proposed wall example has direct classical precedent. Smith, Petrova, and
Schaefer's [progressive surface
codec](https://people.tamu.edu/~gpetrova/surface_paper.pdf) stores fitted planes
inside a pruned adaptive octree. Other work uses [planes and quadric
surfaces](https://pmc.ncbi.nlm.nih.gov/articles/PMC8507489/) or [polygon
clouds](https://www.microsoft.com/en-us/research/publication/dynamic-polygon-cloud-compression/).

Four corners representing one rectangular plane are therefore best understood
as a clean baseline and representation contract, not by itself a novel codec.
The research question is whether a shared decoder can infer a richer vocabulary
of geometry from coordinate relationships without explicit primitive IDs or
feature vectors.

## Closest representation-level prior art

| Work | Bottleneck | Decoder | Difference from strict constellation |
|---|---|---|---|
| [Irregular Convolutional Auto-Encoder](https://arxiv.org/abs/1910.02686) | Sparse latent points plus per-point features | Local conditional point generator | Coordinates are not the whole message; the paper's illustrated latent cloud includes feature channels |
| [SampleNet](https://arxiv.org/abs/1912.03663) | Learned sampled coordinates | Downstream task/reconstructor | Sampling method, not a rate-distortion codec with a shared generative geometry decoder |
| [Feature-Preserving Simplification](https://arxiv.org/abs/2109.14982) | Selected/repositioned points | Conventional surface interpretation | Closest coordinate-only baseline, but no constellation code semantics or bitstream |
| [Sparse Latent Point Diffusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Lyu_Controllable_Mesh_Generation_Through_Sparse_Latent_Point_Diffusion_Models_CVPR_2023_paper.pdf) | Sparse point positions and features | Learned progressive upsampler | Retains feature channels and targets generation |
| Plane/quadric codecs | Explicit primitive parameters and partition structure | Analytic surface sampler | Geometry is typed explicitly and requires structural metadata |

## About “SuperPoint”

The name is ambiguous. In 3D literature, [Superpoint
Network](https://openaccess.thecvf.com/content/ICCV2021/papers/Hui_Superpoint_Network_for_Point_Cloud_Oversegmentation_ICCV_2021_paper.pdf)
and [Superpoint Graphs](https://arxiv.org/abs/1711.09869) partition clouds into
geometrically homogeneous regions for segmentation. They can be useful as a
front-end or baseline partitioner, but they are not themselves geometry
compression codecs. The 2D SuperPoint keypoint detector is unrelated.

## What could be genuinely useful

- **Semantic spatial latent:** constellation points remain editable,
  transformable, and visualizable in the same frame as the source.
- **Continuous level of detail:** the same constellation can be decoded at
  different sample densities.
- **Strong priors for built environments:** walls, floors, pipes, and repeated
  manufactured structures may admit extreme compression.
- **Compressed-domain tasks:** a small geometric set could support detection,
  registration, or planning without dense reconstruction.

## Principal failure modes

1. **No rate advantage:** three quantized coordinates per anchor may cost more
   than an entropy-coded occupancy tree at the same distortion.
2. **Feature smuggling:** floating-point offsets become a hidden feature vector.
3. **Dataset memorization:** a powerful shared decoder reconstructs familiar
   shapes but fails on novel geometry.
4. **Topology ambiguity:** the same unordered anchors can support several
   plausible surfaces.
5. **Density ambiguity:** pure geometry does not communicate sensor sampling
   density unless density is fixed, inferred, or separately coded.
6. **Boundary/detail loss:** anchors favor broad surfaces and miss thin or rare
   structures.
