"""Competitive, decoder-conditioned refinement of coordinate constellations."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from pointconstellation.losses import chamfer_squared, chamfer_squared_chunked
from pointconstellation.quantization import quantize_coordinates, quantize_ste

Decoder = Callable[..., Tensor]


def _validate_points(points: Tensor) -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (batch, N, 3)")


def _fps_initialization(points: Tensor, constellation_size: int) -> Tensor:
    """Return a deterministic, permutation-invariant FPS initialization."""

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
    selected = torch.empty(
        (batch_size, constellation_size), dtype=torch.long, device=points.device
    )
    for slot_index in range(constellation_size):
        selected[:, slot_index] = farthest
        anchor = points[batch_indices, farthest][:, None, :]
        distances = ((points - anchor) ** 2).sum(dim=-1)
        minimum_distances = torch.minimum(minimum_distances, distances)
        farthest = minimum_distances.argmax(dim=1)
    return points.gather(1, selected[:, :, None].expand(-1, -1, 3))


class CompetitiveConstellationRefiner(nn.Module):
    """Refine exchangeable coordinate slots with competitive responsibilities.

    The normal forward path returns only the final quantized coordinates. Passing
    ``return_history=True`` exposes the initial state and every recurrent state
    for loss shaping and diagnostics; those tensors are not part of the message.
    """

    def __init__(
        self,
        max_constellation_size: int,
        *,
        bits: int = 12,
        feature_width: int = 96,
        num_heads: int = 4,
        recurrent_steps: int = 4,
        responsibility_temperature: float = 0.2,
        maximum_update: float = 0.1,
        use_decoder_gradient: bool = False,
        decoder_gradient_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if max_constellation_size < 2:
            raise ValueError("max_constellation_size must be at least 2")
        if not 2 <= bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if feature_width < 4 or feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be positive")
        if responsibility_temperature <= 0:
            raise ValueError("responsibility_temperature must be positive")
        if maximum_update <= 0:
            raise ValueError("maximum_update must be positive")
        if decoder_gradient_chunk_size is not None and decoder_gradient_chunk_size < 1:
            raise ValueError("decoder_gradient_chunk_size must be positive")

        self.max_constellation_size = max_constellation_size
        self.bits = bits
        self.feature_width = feature_width
        self.recurrent_steps = recurrent_steps
        self.responsibility_temperature = responsibility_temperature
        self.maximum_update = maximum_update
        self.use_decoder_gradient = use_decoder_gradient
        self.decoder_gradient_chunk_size = decoder_gradient_chunk_size

        self.point_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        self.slot_query = nn.Linear(feature_width, feature_width, bias=False)
        self.point_key = nn.Linear(feature_width, feature_width, bias=False)
        self.point_value = nn.Linear(feature_width, feature_width, bias=False)
        self.evidence_projection = nn.Linear(feature_width, feature_width)
        self.evidence_norm = nn.LayerNorm(feature_width)
        self.slot_attention = nn.MultiheadAttention(
            feature_width, num_heads, dropout=0.0, batch_first=True
        )
        self.slot_norm = nn.LayerNorm(feature_width)
        self.gradient_embedding = nn.Sequential(
            nn.Linear(3, feature_width),
            nn.Tanh(),
            nn.Linear(feature_width, feature_width),
        )
        self.update_head = nn.Sequential(
            nn.Linear(2 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 3),
            nn.Tanh(),
        )

    def _validate_request(self, points: Tensor, constellation_size: int) -> None:
        _validate_points(points)
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        if constellation_size > self.max_constellation_size:
            raise ValueError("constellation_size exceeds the configured maximum")
        if constellation_size > points.shape[1]:
            raise ValueError("constellation_size cannot exceed the input point count")

    def competitive_responsibilities(
        self,
        slot_features: Tensor,
        coordinates: Tensor,
        point_features: Tensor,
        points: Tensor,
    ) -> Tensor:
        """Return ``K x N`` point-to-slot probabilities.

        The softmax is over slots, so every input point has one unit of evidence
        to distribute among all anchors. This creates direct competition rather
        than allowing every slot to independently attend to the same region.
        """

        queries = self.slot_query(slot_features)
        keys = self.point_key(point_features)
        similarity = torch.einsum("bkd,bnd->bkn", queries, keys)
        similarity = similarity / math.sqrt(self.feature_width)
        squared_distance = (
            (coordinates[:, :, None, :] - points[:, None, :, :]) ** 2
        ).sum(dim=-1)
        logits = similarity - squared_distance / self.responsibility_temperature
        return torch.softmax(logits, dim=1)

    def _decoder_coordinate_gradient(
        self,
        coordinates: Tensor,
        decoder: Decoder,
        target: Tensor,
        num_output_points: int | None,
    ) -> Tensor:
        if isinstance(decoder, nn.Module) and any(
            parameter.requires_grad for parameter in decoder.parameters()
        ):
            raise ValueError("decoder-gradient feedback requires a frozen decoder")
        with torch.enable_grad():
            probe = coordinates.detach().requires_grad_(True)
            quantized = quantize_ste(probe, self.bits, training=False, jitter=False)
            reconstruction = decoder(quantized, num_output_points=num_output_points)
            if self.decoder_gradient_chunk_size is None:
                loss = chamfer_squared(reconstruction, target.detach())
            else:
                loss = chamfer_squared_chunked(
                    reconstruction,
                    target.detach(),
                    chunk_size=self.decoder_gradient_chunk_size,
                )
            gradient = torch.autograd.grad(loss, probe, create_graph=False)[0]
        gradient = gradient.detach()
        scale = gradient.square().mean(dim=(1, 2), keepdim=True).sqrt()
        return gradient / scale.clamp_min(1e-8)

    def forward(
        self,
        points: Tensor,
        constellation_size: int,
        *,
        steps: int | None = None,
        initial_constellation: Tensor | None = None,
        decoder: Decoder | None = None,
        target: Tensor | None = None,
        num_output_points: int | None = None,
        return_history: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        self._validate_request(points, constellation_size)
        step_count = self.recurrent_steps if steps is None else steps
        if step_count < 1 or step_count > self.recurrent_steps:
            raise ValueError("steps must be between 1 and recurrent_steps")
        if self.use_decoder_gradient and (decoder is None or target is None):
            raise ValueError(
                "decoder and target are required when decoder gradients are enabled"
            )
        if target is not None:
            _validate_points(target)
            if target.shape[0] != points.shape[0]:
                raise ValueError("target batch size must match points")

        if initial_constellation is None:
            coordinates = _fps_initialization(points, constellation_size)
        else:
            expected = (len(points), constellation_size, 3)
            if initial_constellation.shape != expected:
                raise ValueError(f"initial_constellation must have shape {expected}")
            coordinates = initial_constellation
        coordinates = quantize_ste(
            coordinates, self.bits, training=self.training, jitter=False
        )
        history = [coordinates]
        point_features = self.point_embedding(points)
        slot_features = self.coordinate_embedding(coordinates)

        for _ in range(step_count):
            responsibilities = self.competitive_responsibilities(
                slot_features, coordinates, point_features, points
            )
            normalized = responsibilities / responsibilities.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-8)
            evidence = torch.einsum(
                "bkn,bnd->bkd", normalized, self.point_value(point_features)
            )
            slot_features = self.evidence_norm(
                slot_features + self.evidence_projection(evidence)
            )
            coordinated, _ = self.slot_attention(
                slot_features, slot_features, slot_features, need_weights=False
            )
            slot_features = self.slot_norm(slot_features + coordinated)

            if self.use_decoder_gradient:
                assert decoder is not None and target is not None
                decoder_gradient = self._decoder_coordinate_gradient(
                    coordinates, decoder, target, num_output_points
                )
                gradient_features = self.gradient_embedding(decoder_gradient)
            else:
                gradient_features = torch.zeros_like(slot_features)
            update = self.update_head(
                torch.cat((slot_features, gradient_features), dim=-1)
            )
            coordinates = (coordinates + self.maximum_update * update).clamp(-1.0, 1.0)
            coordinates = quantize_ste(
                coordinates, self.bits, training=self.training, jitter=False
            )
            history.append(coordinates)
            slot_features = slot_features + self.coordinate_embedding(coordinates)

        if return_history:
            return coordinates, history
        return coordinates

    @torch.no_grad()
    def project_unique_to_input(self, constellation: Tensor, points: Tensor) -> Tensor:
        """Project every slot to a distinct nearest input index.

        Pairs are selected globally from nearest to farthest. The result is a
        strict subset of the quantized input cloud and is equivariant to slot and
        input permutations except at exact distance ties.
        """

        _validate_points(points)
        _validate_points(constellation)
        if constellation.shape[0] != points.shape[0]:
            raise ValueError("constellation and points batch sizes must match")
        if constellation.shape[1] > points.shape[1]:
            raise ValueError("constellation cannot contain more points than input")

        batch_size, constellation_size, _ = constellation.shape
        num_points = points.shape[1]
        distances = ((constellation[:, :, None, :] - points[:, None, :, :]) ** 2).sum(
            dim=-1
        )
        available_slots = torch.ones(
            (batch_size, constellation_size),
            dtype=torch.bool,
            device=points.device,
        )
        available_points = torch.ones(
            (batch_size, num_points), dtype=torch.bool, device=points.device
        )
        selected = torch.empty(
            (batch_size, constellation_size), dtype=torch.long, device=points.device
        )
        batch_indices = torch.arange(batch_size, device=points.device)
        for _ in range(constellation_size):
            available = available_slots[:, :, None] & available_points[:, None, :]
            flat = distances.masked_fill(~available, torch.inf).flatten(1).argmin(1)
            slot_indices = torch.div(flat, num_points, rounding_mode="floor")
            point_indices = flat % num_points
            selected[batch_indices, slot_indices] = point_indices
            available_slots[batch_indices, slot_indices] = False
            available_points[batch_indices, point_indices] = False

        projected = points.gather(1, selected[:, :, None].expand(-1, -1, 3))
        return quantize_coordinates(projected, self.bits)
