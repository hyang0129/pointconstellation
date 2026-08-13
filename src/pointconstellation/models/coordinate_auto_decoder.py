"""Literal coordinate codes and a coordinate-only decoder.

This module deliberately keeps the representation contract small: the complete
message presented to :class:`CoordinateOnlyDecoder` is an unordered ``K x 3``
tensor.  In particular, cloud identifiers and encoder features never cross the
decoder boundary.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pointconstellation.losses import pairwise_squared
from pointconstellation.quantization import quantize_coordinates, quantize_ste

COORDINATE_MODES = ("unrestricted", "projected")


def quantize_constellation(
    constellation: Tensor,
    bits: int,
    *,
    straight_through: bool,
    jitter: bool = False,
) -> Tensor:
    """Apply exact lattice quantization or its straight-through estimator."""

    if straight_through:
        return quantize_ste(
            constellation,
            bits,
            training=True,
            jitter=jitter,
        )
    return quantize_coordinates(constellation, bits)


def project_to_input_surface(
    constellation: Tensor,
    points: Tensor,
    *,
    straight_through: bool,
) -> Tensor:
    """Project every coordinate to its nearest observed input point.

    ``constellation`` may be ``(B, K, 3)`` or ``(B, R, K, 3)``.  Projection is
    exact in the forward pass.  The straight-through form supplies an identity
    gradient to the unprojected coordinate, which is useful for local search.
    """

    if constellation.ndim not in (3, 4) or constellation.shape[-1] != 3:
        raise ValueError("constellation must have shape (B, K, 3) or (B, R, K, 3)")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (B, N, 3)")
    if len(constellation) != len(points):
        raise ValueError("constellation and points must have the same batch size")

    if constellation.ndim == 3:
        flat = constellation
        references = points
    else:
        batch_size, restarts, constellation_size, _ = constellation.shape
        flat = constellation.reshape(batch_size * restarts, constellation_size, 3)
        references = (
            points[:, None]
            .expand(-1, restarts, -1, -1)
            .reshape(batch_size * restarts, points.shape[1], 3)
        )
    indices = pairwise_squared(flat, references).argmin(dim=-1)
    nearest = references.gather(1, indices[:, :, None].expand(-1, -1, 3))
    if constellation.ndim == 4:
        nearest = nearest.reshape_as(constellation)
    if straight_through:
        return constellation + (nearest - constellation).detach()
    return nearest


class LiteralCoordinateBank(nn.Module):
    """A table of independently learnable literal coordinate constellations."""

    def __init__(
        self,
        num_clouds: int,
        num_restarts: int,
        constellation_size: int,
        *,
        bits: int = 12,
        mode: str = "unrestricted",
        initial_clouds: Tensor | None = None,
        seed: int = 7,
    ) -> None:
        super().__init__()
        if num_clouds < 1 or num_restarts < 1:
            raise ValueError("num_clouds and num_restarts must be positive")
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        if mode not in COORDINATE_MODES:
            raise ValueError(f"mode must be one of {COORDINATE_MODES}")
        # Validate the bit depth immediately.
        quantize_coordinates(torch.zeros(1), bits)
        if initial_clouds is not None:
            if initial_clouds.ndim != 3 or initial_clouds.shape[-1] != 3:
                raise ValueError("initial_clouds must have shape (C, N, 3)")
            if len(initial_clouds) != num_clouds:
                raise ValueError("initial_clouds must contain num_clouds clouds")
            if initial_clouds.shape[1] < constellation_size:
                raise ValueError("a cloud cannot be smaller than the constellation")

        self.num_clouds = num_clouds
        self.num_restarts = num_restarts
        self.constellation_size = constellation_size
        self.bits = bits
        self.mode = mode
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if initial_clouds is None:
            initial = torch.empty(
                num_clouds,
                num_restarts,
                constellation_size,
                3,
            ).uniform_(-0.75, 0.75, generator=generator)
        else:
            initial_clouds = initial_clouds.detach().cpu()
            initial = torch.empty(
                num_clouds,
                num_restarts,
                constellation_size,
                3,
                dtype=initial_clouds.dtype,
            )
            for cloud_index in range(num_clouds):
                for restart_index in range(num_restarts):
                    permutation = torch.randperm(
                        initial_clouds.shape[1], generator=generator
                    )
                    initial[cloud_index, restart_index] = initial_clouds[
                        cloud_index, permutation[:constellation_size]
                    ]
            if mode == "unrestricted":
                initial.add_(
                    0.01
                    * torch.randn(
                        initial.shape,
                        dtype=initial.dtype,
                        generator=generator,
                    )
                ).clamp_(-1.0, 1.0)
        # This parameter is the code itself, not a feature vector or codebook ID.
        self.coordinates = nn.Parameter(initial)

    def forward(
        self,
        cloud_indices: Tensor,
        *,
        reference_points: Tensor | None = None,
        straight_through: bool = True,
        jitter: bool = False,
    ) -> Tensor:
        if cloud_indices.ndim != 1:
            raise ValueError("cloud_indices must be a one-dimensional tensor")
        coordinates = self.coordinates[cloud_indices]
        # Clamping is exact forward / identity backward, just like the lattice STE.
        coordinates = (
            coordinates + (coordinates.clamp(-1.0, 1.0) - coordinates).detach()
        )
        if self.mode == "projected":
            if reference_points is None:
                raise ValueError("projected mode requires reference_points")
            coordinates = project_to_input_surface(
                coordinates,
                reference_points,
                straight_through=straight_through,
            )
        return quantize_constellation(
            coordinates,
            self.bits,
            straight_through=straight_through,
            jitter=jitter,
        )


def _folding_grid(num_points: int) -> Tensor:
    width = math.ceil(math.sqrt(num_points))
    axis = torch.linspace(-1.0, 1.0, width)
    first, second = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((first.flatten(), second.flatten()), dim=-1)[:num_points]


class CoordinateOnlyDecoder(nn.Module):
    """Decode an unordered literal coordinate set without hidden side channels."""

    def __init__(self, num_output_points: int, *, feature_width: int = 48) -> None:
        super().__init__()
        if num_output_points < 8:
            raise ValueError("num_output_points must be at least 8")
        if feature_width < 8:
            raise ValueError("feature_width must be at least 8")
        self.num_output_points = num_output_points
        self.feature_width = feature_width
        self.anchor_features = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * feature_width + 2, 2 * feature_width),
            nn.GELU(),
            nn.Linear(2 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
            nn.Tanh(),
        )
        self.register_buffer("queries", _folding_grid(num_output_points))

    def forward(self, constellation: Tensor) -> Tensor:
        if constellation.ndim != 3 or constellation.shape[-1] != 3:
            raise ValueError("constellation must have shape (B, K, 3)")
        features = self.anchor_features(constellation)
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
        pooled = pooled[:, None].expand(-1, self.num_output_points, -1)
        queries = self.queries[None].expand(len(constellation), -1, -1)
        return self.output(torch.cat((pooled, queries), dim=-1))


class CoordinateAutoDecoder(nn.Module):
    """Pair literal per-cloud coordinate codes with a coordinate-only decoder."""

    def __init__(
        self,
        *,
        num_clouds: int,
        num_restarts: int,
        num_output_points: int,
        constellation_size: int,
        bits: int = 12,
        mode: str = "unrestricted",
        feature_width: int = 48,
        initial_clouds: Tensor | None = None,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.codes = LiteralCoordinateBank(
            num_clouds,
            num_restarts,
            constellation_size,
            bits=bits,
            mode=mode,
            initial_clouds=initial_clouds,
            seed=seed,
        )
        self.decoder = CoordinateOnlyDecoder(
            num_output_points,
            feature_width=feature_width,
        )

    def forward(
        self,
        cloud_indices: Tensor,
        *,
        reference_points: Tensor | None = None,
        straight_through: bool = True,
        jitter: bool = False,
    ) -> tuple[Tensor, Tensor]:
        codes = self.codes(
            cloud_indices,
            reference_points=reference_points,
            straight_through=straight_through,
            jitter=jitter,
        )
        batch_size, restarts, constellation_size, _ = codes.shape
        reconstruction = self.decoder(codes.flatten(0, 1)).reshape(
            batch_size,
            restarts,
            self.decoder.num_output_points,
            3,
        )
        if codes.shape != (batch_size, restarts, constellation_size, 3):
            raise RuntimeError("literal coordinate bank violated the K x 3 contract")
        return reconstruction, codes


class PermutationInvariantAmortizer(nn.Module):
    """One-shot set encoder used to initialize held-out coordinate inference."""

    def __init__(self, constellation_size: int, *, feature_width: int = 48) -> None:
        super().__init__()
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        self.constellation_size = constellation_size
        self.point_features = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
            nn.GELU(),
        )
        self.coordinate_head = nn.Sequential(
            nn.Linear(2 * feature_width, 2 * feature_width),
            nn.GELU(),
            nn.Linear(2 * feature_width, 3 * constellation_size),
            nn.Tanh(),
        )

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (B, N, 3)")
        features = self.point_features(points)
        pooled = torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=-1)
        return self.coordinate_head(pooled).reshape(
            len(points), self.constellation_size, 3
        )
