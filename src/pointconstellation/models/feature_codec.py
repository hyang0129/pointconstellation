"""A matched PointNet-style learned feature-latent point-cloud codec."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pointconstellation.quantization import quantize_ste


class VariableFeatureEncoder(nn.Module):
    """Encode an unordered point set into a quantized ordered feature prefix."""

    def __init__(
        self,
        max_latent_dim: int,
        *,
        bits: int = 8,
        feature_width: int = 64,
    ) -> None:
        super().__init__()
        if max_latent_dim < 2:
            raise ValueError("max_latent_dim must be at least 2")
        self.max_latent_dim = max_latent_dim
        self.bits = bits
        self.point_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
            nn.GELU(),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(2 * feature_width, 2 * feature_width),
            nn.GELU(),
            nn.Linear(2 * feature_width, max_latent_dim),
            nn.Tanh(),
        )

    def forward(self, points: Tensor, latent_dim: int) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        if not 1 <= latent_dim <= self.max_latent_dim:
            raise ValueError("latent_dim exceeds the configured range")
        embedded = self.point_embedding(points)
        pooled = torch.cat((embedded.mean(dim=1), embedded.amax(dim=1)), dim=-1)
        features = self.feature_head(pooled)[:, :latent_dim]
        return quantize_ste(
            features,
            self.bits,
            training=self.training,
            jitter=True,
        )


class VariableFeatureDecoder(nn.Module):
    """Decode a variable-length ordered latent into a fixed-size point set."""

    def __init__(
        self,
        max_output_points: int,
        max_latent_dim: int,
        *,
        feature_width: int = 64,
    ) -> None:
        super().__init__()
        if max_output_points < 8:
            raise ValueError("max_output_points must be at least 8")
        if max_latent_dim < 2:
            raise ValueError("max_latent_dim must be at least 2")
        self.max_output_points = max_output_points
        self.max_latent_dim = max_latent_dim
        self.latent_projection = nn.Sequential(
            nn.Linear(max_latent_dim, 2 * feature_width),
            nn.GELU(),
            nn.Linear(2 * feature_width, 2 * feature_width),
        )
        self.cardinality_embedding = nn.Sequential(
            nn.Linear(1, 2 * feature_width),
            nn.GELU(),
            nn.Linear(2 * feature_width, 2 * feature_width),
        )
        self.output_queries = nn.Parameter(
            torch.empty(1, max_output_points, feature_width)
        )
        nn.init.normal_(self.output_queries, std=0.05)
        self.output_head = nn.Sequential(
            nn.Linear(3 * feature_width, 2 * feature_width),
            nn.GELU(),
            nn.Linear(2 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
            nn.Tanh(),
        )

    def forward(
        self,
        features: Tensor,
        *,
        num_output_points: int | None = None,
    ) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, latent_dim)")
        latent_dim = features.shape[1]
        if not 1 <= latent_dim <= self.max_latent_dim:
            raise ValueError("latent_dim exceeds the configured range")
        output_points = num_output_points or self.max_output_points
        if not 1 <= output_points <= self.max_output_points:
            raise ValueError("num_output_points exceeds the configured range")
        padded = torch.nn.functional.pad(
            features, (0, self.max_latent_dim - latent_dim)
        )
        context = self.latent_projection(padded)
        cardinality = features.new_full(
            (len(features), 1), latent_dim / self.max_latent_dim
        )
        context = context + self.cardinality_embedding(cardinality)
        queries = self.output_queries[:, :output_points].expand(len(features), -1, -1)
        return self.output_head(
            torch.cat((queries, context[:, None].expand(-1, output_points, -1)), -1)
        )


class VariableFeatureCodec(nn.Module):
    """Convenience wrapper for joint feature encoder/decoder training."""

    def __init__(
        self,
        max_output_points: int,
        max_latent_dim: int,
        *,
        bits: int = 8,
        feature_width: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = VariableFeatureEncoder(
            max_latent_dim, bits=bits, feature_width=feature_width
        )
        self.decoder = VariableFeatureDecoder(
            max_output_points,
            max_latent_dim,
            feature_width=feature_width,
        )

    def forward(self, points: Tensor, latent_dim: int) -> tuple[Tensor, Tensor]:
        features = self.encoder(points, latent_dim)
        return self.decoder(features, num_output_points=points.shape[1]), features
