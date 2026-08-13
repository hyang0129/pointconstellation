"""Variable-cardinality subset encoder and anchor-preserving decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pointconstellation.quantization import quantize_ste


class ProgressiveSubsetEncoder(nn.Module):
    """Rank input points once, then transmit any requested nested prefix."""

    def __init__(
        self,
        max_constellation_size: int,
        *,
        bits: int = 12,
        feature_width: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        selection_temperature: float = 0.1,
        stochastic_training: bool = False,
        quantization_jitter: bool = True,
    ) -> None:
        super().__init__()
        if max_constellation_size < 2:
            raise ValueError("max_constellation_size must be at least 2")
        if feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        self.max_constellation_size = max_constellation_size
        self.bits = bits
        self.feature_width = feature_width
        self.selection_temperature = selection_temperature
        self.stochastic_training = stochastic_training
        self.quantization_jitter = quantization_jitter
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
        self.score_head = nn.Sequential(
            nn.Linear(3 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 1),
        )

    def importance_scores(self, points: Tensor) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, N, 3)")
        context = self.point_relations(self.point_embedding(points))
        mean_context = context.mean(dim=1, keepdim=True).expand_as(context)
        max_context = context.amax(dim=1, keepdim=True).expand_as(context)
        return self.score_head(
            torch.cat((context, mean_context, max_context), dim=-1)
        ).squeeze(-1)

    def ranked_indices(self, points: Tensor) -> Tensor:
        """Return input indices in descending learned importance order."""

        return self.importance_scores(points).argsort(dim=1, descending=True)

    def forward(self, points: Tensor, constellation_size: int) -> Tensor:
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        if constellation_size > self.max_constellation_size:
            raise ValueError("constellation_size exceeds the configured maximum")
        if constellation_size > points.shape[1]:
            raise ValueError("constellation_size cannot exceed the input point count")

        scores = self.importance_scores(points)
        if not self.training:
            indices = scores.topk(constellation_size, dim=1).indices
            anchors = points.gather(1, indices[:, :, None].expand(-1, -1, 3))
            return quantize_ste(anchors, self.bits, training=False)

        available = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        selected_anchors = []
        for _ in range(constellation_size):
            masked_scores = scores.masked_fill(
                ~available, torch.finfo(scores.dtype).min
            )
            sampling_scores = masked_scores
            if self.stochastic_training:
                uniform = torch.rand_like(masked_scores).clamp_(1e-6, 1.0 - 1e-6)
                sampling_scores = sampling_scores - torch.log(-torch.log(uniform))
            soft_weights = torch.softmax(
                sampling_scores / self.selection_temperature, dim=-1
            )
            selected_indices = sampling_scores.argmax(dim=-1, keepdim=True)
            hard_weights = torch.zeros_like(soft_weights).scatter(
                1, selected_indices, 1.0
            )
            weights = hard_weights + soft_weights - soft_weights.detach()
            selected_anchors.append(torch.einsum("bn,bnd->bd", weights, points))
            available = available.scatter(1, selected_indices, False)

        anchors = torch.stack(selected_anchors, dim=1)
        return quantize_ste(
            anchors,
            self.bits,
            training=True,
            jitter=self.quantization_jitter,
        )


class VariableConstellationDecoder(nn.Module):
    """Complete variable-size coordinate sets while retaining every anchor."""

    def __init__(
        self,
        max_output_points: int,
        max_constellation_size: int,
        *,
        feature_width: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if max_output_points < 8:
            raise ValueError("max_output_points must be at least 8")
        if max_constellation_size < 2:
            raise ValueError("max_constellation_size must be at least 2")
        if max_constellation_size > max_output_points:
            raise ValueError("max_constellation_size cannot exceed max_output_points")
        if feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        self.max_output_points = max_output_points
        self.max_constellation_size = max_constellation_size
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
            torch.empty(1, max_output_points, feature_width)
        )
        nn.init.normal_(self.output_queries, std=0.02)
        self.cardinality_embedding = nn.Sequential(
            nn.Linear(1, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
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

    def forward(
        self, constellation: Tensor, *, num_output_points: int | None = None
    ) -> Tensor:
        if constellation.ndim != 3 or constellation.shape[-1] != 3:
            raise ValueError("constellation must have shape (batch, K, 3)")
        constellation_size = constellation.shape[1]
        if constellation_size > self.max_constellation_size:
            raise ValueError("constellation exceeds the configured maximum")
        output_size = num_output_points or self.max_output_points
        if output_size > self.max_output_points:
            raise ValueError("num_output_points exceeds the configured maximum")
        if output_size < constellation_size:
            raise ValueError(
                "num_output_points cannot be smaller than the constellation"
            )
        generated_count = output_size - constellation_size
        if not generated_count:
            return constellation

        context = self.anchor_relations(self.anchor_embedding(constellation))
        queries = self.output_queries[:, :generated_count].expand(
            len(constellation), -1, -1
        )
        cardinality = constellation.new_full(
            (len(constellation), 1),
            constellation_size / self.max_constellation_size,
        )
        queries = queries + self.cardinality_embedding(cardinality)[:, None, :]
        attended, _ = self.output_attention(
            queries, context, context, need_weights=False
        )
        generated = self.output_head(self.output_norm(queries + attended))
        return torch.cat((constellation, generated), dim=1)
