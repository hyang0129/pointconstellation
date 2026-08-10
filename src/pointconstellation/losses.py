"""Differentiable geometry losses for constellation training."""

from __future__ import annotations

import torch
from torch import Tensor


def pairwise_squared(first: Tensor, second: Tensor) -> Tensor:
    """Return batched squared Euclidean distances without ``torch.cdist``."""

    first_norm = (first**2).sum(dim=-1, keepdim=True)
    second_norm = (second**2).sum(dim=-1).unsqueeze(1)
    product = torch.bmm(first, second.transpose(1, 2))
    return (first_norm + second_norm - 2.0 * product).clamp_min(0.0)


def chamfer_squared(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean() + distances.amin(dim=1).mean())


def anchor_surface_loss(constellation: Tensor, points: Tensor) -> Tensor:
    return pairwise_squared(constellation, points).amin(dim=2).mean()


def repulsion_loss(constellation: Tensor, *, margin: float = 0.1) -> Tensor:
    distances = pairwise_squared(constellation, constellation)
    count = constellation.shape[1]
    diagonal = torch.eye(count, dtype=torch.bool, device=constellation.device)[None]
    distances = distances.masked_fill(diagonal, margin**2)
    return torch.relu(margin**2 - distances).mean()


def constellation_loss(
    reconstruction: Tensor,
    target: Tensor,
    constellation: Tensor,
    *,
    surface_weight: float = 0.1,
    repulsion_weight: float = 0.01,
) -> tuple[Tensor, dict[str, Tensor]]:
    chamfer = chamfer_squared(reconstruction, target)
    surface = anchor_surface_loss(constellation, target)
    repulsion = repulsion_loss(constellation)
    total = chamfer + surface_weight * surface + repulsion_weight * repulsion
    return total, {
        "loss": total.detach(),
        "chamfer": chamfer.detach(),
        "surface": surface.detach(),
        "repulsion": repulsion.detach(),
    }
