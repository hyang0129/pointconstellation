"""Learned coordinate-only compression along a decreasing-cardinality path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from pointconstellation.quantization import quantize_ste


def _validate_points(points: Tensor, *, name: str = "points") -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (batch, N, 3)")


def _validate_stage_sizes(stage_sizes: tuple[int, ...]) -> None:
    if len(stage_sizes) < 2:
        raise ValueError("stage_sizes must contain at least two sizes")
    if any(size < 2 for size in stage_sizes):
        raise ValueError("every stage size must be at least 2")
    if any(
        first <= second
        for first, second in zip(stage_sizes, stage_sizes[1:], strict=False)
    ):
        raise ValueError("stage_sizes must be strictly decreasing")


def farthest_point_indices(points: Tensor, count: int) -> Tensor:
    """Select deterministic FPS indices without depending on input order.

    The first point is farthest from the set centroid. Subsequent points are
    farthest from the selected set. As with every argmax implementation, exact
    geometric ties may depend on storage order.
    """

    _validate_points(points)
    if count < 2:
        raise ValueError("count must be at least 2")
    if count > points.shape[1]:
        raise ValueError("count cannot exceed the input point count")

    batch_size, num_points, _ = points.shape
    rows = torch.arange(batch_size, device=points.device)
    centroid = points.mean(dim=1, keepdim=True)
    farthest = ((points - centroid) ** 2).sum(dim=-1).argmax(dim=1)
    minimum = torch.full(
        (batch_size, num_points),
        torch.inf,
        dtype=points.dtype,
        device=points.device,
    )
    selected = torch.empty((batch_size, count), dtype=torch.long, device=points.device)
    for index in range(count):
        selected[:, index] = farthest
        anchor = points[rows, farthest][:, None, :]
        minimum = torch.minimum(minimum, ((points - anchor) ** 2).sum(dim=-1))
        farthest = minimum.argmax(dim=1)
    return selected


def farthest_point_constellation(
    points: Tensor,
    count: int,
    bits: int,
    *,
    training: bool,
) -> Tensor:
    """Return a coordinate-only, exactly quantized FPS constellation."""

    indices = farthest_point_indices(points, count)
    selected = points.gather(1, indices[:, :, None].expand(-1, -1, 3))
    return quantize_ste(selected, bits, training=training, jitter=False)


@dataclass(frozen=True)
class MergeDiagnostics:
    """Differentiable transition diagnostics that never cross the decoder API."""

    soft_assignments: Tensor
    hard_assignments: Tensor
    merge_weights: Tensor


class ConditionalMergeTransition(nn.Module):
    """Conditionally merge a current set into a smaller coordinate set.

    Current anchors compete for target groups. The forward pass uses discrete
    group assignments while its gradient follows a soft relaxation. Each group
    is seeded by FPS, preventing empty hard groups. A learned, bounded coordinate
    update then moves each merged coordinate before exact lattice quantization.
    """

    def __init__(
        self,
        *,
        bits: int = 12,
        feature_width: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        merge_temperature: float = 0.2,
        maximum_update: float = 0.08,
    ) -> None:
        super().__init__()
        if not 2 <= bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if feature_width < 4 or feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if merge_temperature <= 0:
            raise ValueError("merge_temperature must be positive")
        if maximum_update <= 0:
            raise ValueError("maximum_update must be positive")

        self.bits = bits
        self.feature_width = feature_width
        self.merge_temperature = merge_temperature
        self.maximum_update = maximum_update
        self.coordinate_embedding = nn.Sequential(
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
        self.relations = nn.TransformerEncoder(
            relation_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.query = nn.Linear(feature_width, feature_width, bias=False)
        self.key = nn.Linear(feature_width, feature_width, bias=False)
        self.value = nn.Linear(feature_width, feature_width, bias=False)
        self.update_head = nn.Sequential(
            nn.Linear(2 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
            nn.Tanh(),
        )

    def _assignments(
        self,
        current: Tensor,
        context: Tensor,
        target_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        seed_indices = farthest_point_indices(current, target_size)
        seed_coordinates = current.gather(1, seed_indices[:, :, None].expand(-1, -1, 3))
        seed_features = context.gather(
            1,
            seed_indices[:, :, None].expand(-1, -1, self.feature_width),
        )
        logits = torch.einsum(
            "bkd,bnd->bkn", self.query(seed_features), self.key(context)
        ) / math.sqrt(self.feature_width)
        squared_distance = (
            (seed_coordinates[:, :, None, :] - current[:, None, :, :]) ** 2
        ).sum(dim=-1)
        logits = logits - squared_distance / self.merge_temperature

        # Each current coordinate distributes one unit of mass over target groups.
        soft_assignments = torch.softmax(logits, dim=1)
        winners = logits.argmax(dim=1)
        hard_assignments = functional.one_hot(
            winners, num_classes=target_size
        ).transpose(1, 2)
        hard_assignments = hard_assignments.to(dtype=current.dtype)

        # Force each FPS seed into its corresponding group. This is only a
        # forward constraint; the relaxation below still supplies smooth grads.
        seed_membership = functional.one_hot(
            seed_indices, num_classes=current.shape[1]
        ).to(dtype=current.dtype)
        seed_columns = seed_membership.amax(dim=1, keepdim=True)
        hard_assignments = hard_assignments * (1.0 - seed_columns) + seed_membership

        hard_weights = hard_assignments / hard_assignments.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0)
        soft_weights = soft_assignments / soft_assignments.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-8)
        merge_weights = hard_weights + soft_weights - soft_weights.detach()
        return soft_assignments, hard_assignments, merge_weights

    def forward(
        self,
        current: Tensor,
        target_size: int,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, MergeDiagnostics]:
        _validate_points(current, name="current")
        if target_size < 2:
            raise ValueError("target_size must be at least 2")
        if target_size >= current.shape[1]:
            raise ValueError("target_size must be smaller than the current set")

        context = self.relations(self.coordinate_embedding(current))
        soft, hard, merge_weights = self._assignments(current, context, target_size)
        merged_coordinates = torch.einsum("bkn,bnd->bkd", merge_weights, current)
        merged_features = torch.einsum(
            "bkn,bnd->bkd", merge_weights, self.value(context)
        )
        global_features = context.mean(dim=1, keepdim=True).expand_as(merged_features)
        update = self.maximum_update * self.update_head(
            torch.cat((merged_features, global_features), dim=-1)
        )
        coordinates = (merged_coordinates + update).clamp(-1.0, 1.0)
        coordinates = quantize_ste(
            coordinates,
            self.bits,
            training=self.training,
            jitter=False,
        )
        if return_diagnostics:
            return coordinates, MergeDiagnostics(soft, hard, merge_weights)
        return coordinates


class CompressionHomotopyEncoder(nn.Module):
    """Encode a cloud by progressively compressing a dense FPS constellation.

    The message at every stage consists solely of ``K x 3`` quantized
    coordinates. The same transition parameters are reused at every stage so a
    direct ``K_dense -> K_target`` baseline can start from identical parameters.
    """

    def __init__(
        self,
        stage_sizes: tuple[int, ...],
        *,
        bits: int = 12,
        feature_width: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        merge_temperature: float = 0.2,
        maximum_update: float = 0.08,
    ) -> None:
        super().__init__()
        _validate_stage_sizes(stage_sizes)
        self.stage_sizes = stage_sizes
        self.bits = bits
        self.transition = ConditionalMergeTransition(
            bits=bits,
            feature_width=feature_width,
            num_heads=num_heads,
            num_layers=num_layers,
            merge_temperature=merge_temperature,
            maximum_update=maximum_update,
        )

    def forward(
        self,
        points: Tensor,
        *,
        stage_sizes: tuple[int, ...] | None = None,
        return_history: bool = False,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor], list[MergeDiagnostics]]:
        _validate_points(points)
        requested = self.stage_sizes if stage_sizes is None else stage_sizes
        _validate_stage_sizes(requested)
        if requested[0] != self.stage_sizes[0]:
            raise ValueError("requested path must use the configured dense stage")
        if requested[0] > points.shape[1]:
            raise ValueError("dense stage cannot exceed the input point count")

        current = farthest_point_constellation(
            points,
            requested[0],
            self.bits,
            training=self.training,
        )
        history = [current]
        diagnostics: list[MergeDiagnostics] = []
        for target_size in requested[1:]:
            transitioned = self.transition(
                current,
                target_size,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                assert isinstance(transitioned, tuple)
                current, stage_diagnostics = transitioned
                diagnostics.append(stage_diagnostics)
            else:
                assert isinstance(transitioned, Tensor)
                current = transitioned
            history.append(current)

        if return_history or return_diagnostics:
            return current, history, diagnostics
        return current
