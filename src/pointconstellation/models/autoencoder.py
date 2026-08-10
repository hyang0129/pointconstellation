"""A minimal learned point-cloud autoencoder with a strict coordinate bottleneck."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pointconstellation.quantization import quantize_ste


class PointwiseMLP(nn.Module):
    def __init__(self, dimensions: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for input_size, output_size in zip(
            dimensions[:-1], dimensions[1:], strict=True
        ):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        self.network = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        return self.network(points)


class ConstellationEncoder(nn.Module):
    """Map an unordered cloud to quantized, surface-proximal 3D anchors."""

    def __init__(
        self,
        constellation_size: int,
        *,
        bits: int = 12,
        projection_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        self.constellation_size = constellation_size
        self.bits = bits
        self.projection_temperature = projection_temperature
        self.point_features = PointwiseMLP((3, 64, 128, 256))
        self.proposals = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, constellation_size * 3),
            nn.Tanh(),
        )

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        features = self.point_features(points)
        global_features = features.amax(dim=1)
        proposals = self.proposals(global_features).reshape(
            len(points), self.constellation_size, 3
        )

        squared = ((proposals[:, :, None, :] - points[:, None, :, :]) ** 2).sum(dim=-1)
        weights = torch.softmax(-squared / self.projection_temperature, dim=-1)
        anchors = torch.einsum("bkn,bnd->bkd", weights, points)
        return quantize_ste(anchors, self.bits, training=self.training)


class FarthestPointEncoder(nn.Module):
    """Select a deterministic FPS constellation with the same quantizer."""

    def __init__(self, constellation_size: int, *, bits: int = 12) -> None:
        super().__init__()
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        self.constellation_size = constellation_size
        self.bits = bits

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        if self.constellation_size > points.shape[1]:
            raise ValueError("constellation_size cannot exceed the input point count")

        batch_size, num_points, _ = points.shape
        batch_indices = torch.arange(batch_size, device=points.device)
        centroid = points.mean(dim=1, keepdim=True)
        farthest = ((points - centroid) ** 2).sum(dim=-1).argmax(dim=1)
        minimum_distances = torch.full(
            (batch_size, num_points),
            torch.inf,
            dtype=points.dtype,
            device=points.device,
        )
        selected_indices = torch.empty(
            (batch_size, self.constellation_size),
            dtype=torch.long,
            device=points.device,
        )

        for anchor_index in range(self.constellation_size):
            selected_indices[:, anchor_index] = farthest
            selected = points[batch_indices, farthest][:, None, :]
            squared_distances = ((points - selected) ** 2).sum(dim=-1)
            minimum_distances = torch.minimum(minimum_distances, squared_distances)
            farthest = minimum_distances.argmax(dim=1)

        constellation = points.gather(1, selected_indices[:, :, None].expand(-1, -1, 3))
        return quantize_ste(constellation, self.bits, training=self.training)


def _folding_grid(num_points: int) -> Tensor:
    width = math.ceil(math.sqrt(num_points))
    axis = torch.linspace(-1.0, 1.0, width)
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((u.reshape(-1), v.reshape(-1)), dim=1)[:num_points]


class ConstellationDecoder(nn.Module):
    """Generate a dense point set using only constellation coordinates."""

    def __init__(self, num_output_points: int) -> None:
        super().__init__()
        if num_output_points < 8:
            raise ValueError("num_output_points must be at least 8")
        self.num_output_points = num_output_points
        self.geometry_features = PointwiseMLP((3, 64, 128, 256))
        self.fold = nn.Sequential(
            nn.Linear(258, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh(),
        )
        self.register_buffer("queries", _folding_grid(num_output_points))

    def forward(self, constellation: Tensor) -> Tensor:
        if constellation.ndim != 3 or constellation.shape[-1] != 3:
            raise ValueError("constellation must have shape (batch, K, 3)")
        geometry = self.geometry_features(constellation).amax(dim=1)
        expanded_geometry = geometry[:, None, :].expand(-1, self.num_output_points, -1)
        queries = self.queries[None, :, :].expand(len(constellation), -1, -1)
        return self.fold(torch.cat((expanded_geometry, queries), dim=-1))


class RelationAwareConstellationEncoder(nn.Module):
    """Propose anchors from permutation-equivariant point relationships."""

    def __init__(
        self,
        constellation_size: int,
        *,
        bits: int = 12,
        feature_width: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        projection_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        if feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        self.constellation_size = constellation_size
        self.bits = bits
        self.projection_temperature = projection_temperature
        self.point_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=feature_width,
            nhead=num_heads,
            dim_feedforward=2 * feature_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.point_relations = nn.TransformerEncoder(
            relation_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.anchor_queries = nn.Parameter(
            torch.empty(1, constellation_size, feature_width)
        )
        nn.init.normal_(self.anchor_queries, std=0.02)
        self.anchor_attention = nn.MultiheadAttention(
            feature_width, num_heads, dropout=0.0, batch_first=True
        )
        self.anchor_norm = nn.LayerNorm(feature_width)
        self.proposal_head = nn.Sequential(
            nn.Linear(feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
            nn.Tanh(),
        )

    def forward(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        context = self.point_relations(self.point_embedding(points))
        queries = self.anchor_queries.expand(len(points), -1, -1)
        attended, _ = self.anchor_attention(
            queries, context, context, need_weights=False
        )
        proposals = self.proposal_head(self.anchor_norm(queries + attended))

        squared = ((proposals[:, :, None, :] - points[:, None, :, :]) ** 2).sum(dim=-1)
        weights = torch.softmax(-squared / self.projection_temperature, dim=-1)
        anchors = torch.einsum("bkn,bnd->bkd", weights, points)
        return quantize_ste(anchors, self.bits, training=self.training)


class RelationAwareConstellationDecoder(nn.Module):
    """Decode each output query by attending to the full coordinate set."""

    def __init__(
        self,
        num_output_points: int,
        *,
        feature_width: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_output_points < 8:
            raise ValueError("num_output_points must be at least 8")
        if feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        self.num_output_points = num_output_points
        self.anchor_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=feature_width,
            nhead=num_heads,
            dim_feedforward=2 * feature_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.anchor_relations = nn.TransformerEncoder(
            relation_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_queries = nn.Parameter(
            torch.empty(1, num_output_points, feature_width)
        )
        nn.init.normal_(self.output_queries, std=0.02)
        self.output_attention = nn.MultiheadAttention(
            feature_width, num_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(feature_width)
        self.output_head = nn.Sequential(
            nn.Linear(feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
            nn.Tanh(),
        )

    def forward(self, constellation: Tensor) -> Tensor:
        if constellation.ndim != 3 or constellation.shape[-1] != 3:
            raise ValueError("constellation must have shape (batch, K, 3)")
        context = self.anchor_relations(self.anchor_embedding(constellation))
        queries = self.output_queries.expand(len(constellation), -1, -1)
        attended, _ = self.output_attention(
            queries, context, context, need_weights=False
        )
        return self.output_head(self.output_norm(queries + attended))


class ConstellationAutoencoder(nn.Module):
    """End-to-end model whose complete latent message is ``(B, K, 3)``."""

    def __init__(
        self,
        *,
        num_input_points: int,
        constellation_size: int,
        bits: int = 12,
    ) -> None:
        super().__init__()
        # Construct the shared decoder first so resetting the seed gives learned
        # and FPS experiments identical decoder initialization.
        self.decoder = ConstellationDecoder(num_input_points)
        self.encoder = ConstellationEncoder(constellation_size, bits=bits)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        constellation = self.encoder(points)
        if constellation.shape != (len(points), self.encoder.constellation_size, 3):
            raise RuntimeError("encoder violated the strict K x 3 bottleneck")
        reconstruction = self.decoder(constellation)
        return reconstruction, constellation


class FPSAutoencoder(nn.Module):
    """Matched-rate baseline: quantized FPS coordinates plus the same decoder."""

    def __init__(
        self,
        *,
        num_input_points: int,
        constellation_size: int,
        bits: int = 12,
    ) -> None:
        super().__init__()
        self.decoder = ConstellationDecoder(num_input_points)
        self.encoder = FarthestPointEncoder(constellation_size, bits=bits)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        constellation = self.encoder(points)
        if constellation.shape != (len(points), self.encoder.constellation_size, 3):
            raise RuntimeError("encoder violated the strict K x 3 bottleneck")
        reconstruction = self.decoder(constellation)
        return reconstruction, constellation


class RelationAwareConstellationAutoencoder(nn.Module):
    """Relation-aware coordinate-only autoencoder for Experiment 002."""

    def __init__(
        self,
        *,
        num_input_points: int,
        constellation_size: int,
        bits: int = 12,
    ) -> None:
        super().__init__()
        self.decoder = RelationAwareConstellationDecoder(num_input_points)
        self.encoder = RelationAwareConstellationEncoder(constellation_size, bits=bits)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        constellation = self.encoder(points)
        if constellation.shape != (len(points), self.encoder.constellation_size, 3):
            raise RuntimeError("encoder violated the strict K x 3 bottleneck")
        return self.decoder(constellation), constellation


class RelationAwareFPSAutoencoder(nn.Module):
    """Quantized FPS coordinates with the relation-aware decoder."""

    def __init__(
        self,
        *,
        num_input_points: int,
        constellation_size: int,
        bits: int = 12,
    ) -> None:
        super().__init__()
        self.decoder = RelationAwareConstellationDecoder(num_input_points)
        self.encoder = FarthestPointEncoder(constellation_size, bits=bits)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        constellation = self.encoder(points)
        if constellation.shape != (len(points), self.encoder.constellation_size, 3):
            raise RuntimeError("encoder violated the strict K x 3 bottleneck")
        return self.decoder(constellation), constellation
