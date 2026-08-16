"""Memory-bounded geometry metrics for the standardized toy benchmark."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def directed_nearest_squared(
    source: Tensor,
    target: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Return nearest squared distances/indices without an ``N x M`` allocation."""

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must have shape (N, D)")
    if source.shape[1] != target.shape[1] or not len(source) or not len(target):
        raise ValueError("source and target must be nonempty with matching dimensions")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    distances = []
    indices = []
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        squared = torch.cdist(chunk[None], target[None]).squeeze(0).square()
        nearest, nearest_indices = squared.min(dim=1)
        distances.append(nearest)
        indices.append(nearest_indices)
    return torch.cat(distances), torch.cat(indices)


def _sliced_wasserstein_rms(
    reconstruction: Tensor,
    target: Tensor,
    *,
    directions: int,
) -> float:
    """Deterministic sliced-Wasserstein RMS; a cheap proxy, not exact EMD."""

    if directions < 1:
        raise ValueError("directions must be positive")
    index = torch.arange(
        directions, dtype=reconstruction.dtype, device=reconstruction.device
    )
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / directions
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    vectors = torch.stack(
        (
            radius * torch.cos(index * golden_angle),
            radius * torch.sin(index * golden_angle),
            z,
        ),
        dim=1,
    )
    first = (reconstruction @ vectors.T).sort(dim=0).values
    second = (target @ vectors.T).sort(dim=0).values
    if len(first) != len(second):
        positions = torch.linspace(
            0,
            len(second) - 1,
            len(first),
            dtype=second.dtype,
            device=second.device,
        )
        lower = positions.floor().long()
        upper = positions.ceil().long()
        weight = (positions - lower).unsqueeze(1)
        second = second[lower] * (1.0 - weight) + second[upper] * weight
    return float(torch.sqrt((first - second).square().mean()).item())


def standardized_geometry_metrics(
    reconstruction: Tensor,
    target: Tensor,
    target_normals: Tensor,
    *,
    chunk_size: int = 256,
    sliced_directions: int = 32,
    peak_distance: float = 2.0,
) -> dict[str, float]:
    """Compute paper-inspired metrics for one normalized point-cloud pair.

    D1/D2 PSNR use a declared peak distance of two for the normalized
    ``[-1, 1]`` protocol. They are protocol proxies, not official MPEG pc_error
    results. Sliced Wasserstein is likewise reported as an EMD proxy.
    """

    if reconstruction.ndim != 2 or reconstruction.shape[1] != 3:
        raise ValueError("reconstruction must have shape (N, 3)")
    if target.shape != target_normals.shape or target.ndim != 2:
        raise ValueError("target and target_normals must have matching (N, 3) shapes")
    if peak_distance <= 0:
        raise ValueError("peak_distance must be positive")

    forward, forward_indices = directed_nearest_squared(
        reconstruction, target, chunk_size=chunk_size
    )
    backward, backward_indices = directed_nearest_squared(
        target, reconstruction, chunk_size=chunk_size
    )
    forward_mean = forward.mean()
    backward_mean = backward.mean()
    chamfer_mse = 0.5 * (forward_mean + backward_mean)
    d1_mse = torch.maximum(forward_mean, backward_mean)

    forward_delta = reconstruction - target[forward_indices]
    forward_plane = (
        (forward_delta * target_normals[forward_indices]).sum(dim=1).square()
    )
    # The target is the only cloud with analytic normals. For the reverse
    # direction, use the normal attached to each target point and its nearest
    # reconstructed point.
    backward_delta = target - reconstruction[backward_indices]
    backward_plane = (backward_delta * target_normals).sum(dim=1).square()
    d2_mse = torch.maximum(forward_plane.mean(), backward_plane.mean())

    directed_euclidean = torch.cat((forward.sqrt(), backward.sqrt()))
    eps = torch.finfo(chamfer_mse.dtype).tiny
    peak_squared = peak_distance**2

    def psnr(mse: Tensor) -> float:
        if float(mse.item()) == 0.0:
            return float("inf")
        ratio = mse.new_tensor(peak_squared) / mse.clamp_min(eps)
        return float(10.0 * torch.log10(ratio).item())

    return {
        "chamfer_mse": float(chamfer_mse.item()),
        "chamfer_rmse": float(torch.sqrt(chamfer_mse).item()),
        "d1_mse_proxy": float(d1_mse.item()),
        "d1_psnr_db_proxy": psnr(d1_mse),
        "d2_mse_proxy": float(d2_mse.item()),
        "d2_psnr_db_proxy": psnr(d2_mse),
        "p95_euclidean": float(torch.quantile(directed_euclidean, 0.95).item()),
        "p99_euclidean": float(torch.quantile(directed_euclidean, 0.99).item()),
        "hausdorff": float(directed_euclidean.max().item()),
        "sliced_wasserstein_rms_proxy": _sliced_wasserstein_rms(
            reconstruction, target, directions=sliced_directions
        ),
    }
