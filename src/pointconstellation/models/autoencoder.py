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
        self.encoder = ConstellationEncoder(constellation_size, bits=bits)
        self.decoder = ConstellationDecoder(num_input_points)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        constellation = self.encoder(points)
        if constellation.shape != (len(points), self.encoder.constellation_size, 3):
            raise RuntimeError("encoder violated the strict K x 3 bottleneck")
        reconstruction = self.decoder(constellation)
        return reconstruction, constellation
