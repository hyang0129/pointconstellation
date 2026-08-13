"""Autoregressive pointer selection for strict point-cloud subsets."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pointconstellation.losses import pairwise_squared
from pointconstellation.quantization import quantize_coordinates, quantize_ste

Decoder = Callable[..., Tensor]


def _validate_points(points: Tensor, *, name: str = "points") -> None:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (batch, N, 3)")


def _chamfer_per_sample(first: Tensor, second: Tensor) -> Tensor:
    distances = pairwise_squared(first, second)
    return 0.5 * (distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1))


@dataclass(frozen=True)
class PointerTrace:
    """Selection diagnostics that are not part of the transmitted message."""

    indices: Tensor
    entropies: Tensor
    logits: tuple[Tensor, ...]
    probabilities: tuple[Tensor, ...]


@dataclass(frozen=True)
class BeamSearchResult:
    """Decoder-scored completed beam selections."""

    coordinates: Tensor
    indices: Tensor
    losses: Tensor
    greedy_losses: Tensor
    decoder_evaluations: Tensor


class AutoregressivePointerSubsetEncoder(nn.Module):
    """Select unique input coordinates while repeatedly updating coverage state.

    At each turn, the logits use permutation-equivariant point context, a pooled
    representation of the already-selected set, and each candidate's distance
    to its nearest selected coordinate. Requested cardinality is also an input,
    so unrelated requests do not share a compulsory global ranking.
    """

    def __init__(
        self,
        max_constellation_size: int,
        *,
        bits: int = 12,
        feature_width: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        selection_temperature: float = 0.2,
        stochastic_training: bool = True,
    ) -> None:
        super().__init__()
        if max_constellation_size < 2:
            raise ValueError("max_constellation_size must be at least 2")
        if not 2 <= bits <= 24:
            raise ValueError("bits must be between 2 and 24")
        if feature_width < 4 or feature_width % num_heads:
            raise ValueError("feature_width must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")

        self.max_constellation_size = max_constellation_size
        self.bits = bits
        self.feature_width = feature_width
        self.selection_temperature = selection_temperature
        self.stochastic_training = stochastic_training

        # These three modules intentionally match ProgressiveSubsetEncoder so
        # its entire scalar-ranking path can receive a copied initialization.
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
            relation_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.score_head = nn.Sequential(
            nn.Linear(3 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 1),
        )

        self.initial_selected_state = nn.Parameter(torch.zeros(feature_width))
        self.selected_projection = nn.Linear(feature_width, feature_width)
        self.residual_embedding = nn.Sequential(
            nn.Linear(1, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        self.cardinality_embedding = nn.Sequential(
            nn.Linear(1, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, feature_width),
        )
        self.conditional_score = nn.Sequential(
            nn.Linear(4 * feature_width, feature_width),
            nn.GELU(),
            nn.Linear(feature_width, 1),
        )

    def _validate_request(self, points: Tensor, constellation_size: int) -> None:
        _validate_points(points)
        if constellation_size < 2:
            raise ValueError("constellation_size must be at least 2")
        if constellation_size > self.max_constellation_size:
            raise ValueError("constellation_size exceeds the configured maximum")
        if constellation_size > points.shape[1]:
            raise ValueError("constellation_size cannot exceed the input point count")

    def _context(self, points: Tensor) -> tuple[Tensor, Tensor]:
        context = self.point_relations(self.point_embedding(points))
        mean_context = context.mean(dim=1, keepdim=True).expand_as(context)
        max_context = context.amax(dim=1, keepdim=True).expand_as(context)
        base_logits = self.score_head(
            torch.cat((context, mean_context, max_context), dim=-1)
        ).squeeze(-1)
        return context, base_logits

    def _conditional_logits(
        self,
        context: Tensor,
        base_logits: Tensor,
        points: Tensor,
        selected_features: list[Tensor],
        selected_coordinates: list[Tensor],
        constellation_size: int,
    ) -> Tensor:
        batch_size, num_points, _ = points.shape
        if selected_features:
            selected_state = torch.stack(selected_features, dim=1).mean(dim=1)
            selected_points = torch.stack(selected_coordinates, dim=1)
            residual = pairwise_squared(points, selected_points).amin(dim=2)
        else:
            selected_state = self.initial_selected_state[None].expand(batch_size, -1)
            centroid = points.mean(dim=1, keepdim=True)
            residual = ((points - centroid) ** 2).sum(dim=-1)
        residual = residual / residual.mean(dim=1, keepdim=True).clamp_min(1e-8)
        residual_features = self.residual_embedding(residual[:, :, None])
        selected_context = self.selected_projection(selected_state)[:, None].expand(
            -1, num_points, -1
        )
        requested = points.new_full(
            (batch_size, 1),
            constellation_size / self.max_constellation_size,
        )
        requested_context = self.cardinality_embedding(requested)[:, None].expand(
            -1, num_points, -1
        )
        conditional = self.conditional_score(
            torch.cat(
                (context, selected_context, residual_features, requested_context),
                dim=-1,
            )
        ).squeeze(-1)
        return base_logits + conditional

    def conditional_logits(
        self,
        points: Tensor,
        selected_indices: Tensor,
        constellation_size: int,
    ) -> Tensor:
        """Expose next-step logits for diagnostics and beam expansion."""

        self._validate_request(points, constellation_size)
        if selected_indices.ndim != 2 or selected_indices.shape[0] != len(points):
            raise ValueError("selected_indices must have shape (batch, selected)")
        if selected_indices.shape[1] >= constellation_size:
            raise ValueError("selected count must be smaller than constellation_size")
        if selected_indices.numel() and (
            selected_indices.min() < 0 or selected_indices.max() >= points.shape[1]
        ):
            raise ValueError("selected index is out of range")

        context, base_logits = self._context(points)
        selected_features = [
            context.gather(
                1,
                selected_indices[:, index : index + 1, None].expand(
                    -1, -1, self.feature_width
                ),
            ).squeeze(1)
            for index in range(selected_indices.shape[1])
        ]
        selected_coordinates = [
            points.gather(
                1,
                selected_indices[:, index : index + 1, None].expand(-1, -1, 3),
            ).squeeze(1)
            for index in range(selected_indices.shape[1])
        ]
        logits = self._conditional_logits(
            context,
            base_logits,
            points,
            selected_features,
            selected_coordinates,
            constellation_size,
        )
        if selected_indices.numel():
            logits = logits.scatter(
                1,
                selected_indices,
                torch.finfo(logits.dtype).min,
            )
        return logits

    def forward(
        self,
        points: Tensor,
        constellation_size: int,
        *,
        stochastic: bool | None = None,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, PointerTrace]:
        self._validate_request(points, constellation_size)
        stochastic_selection = (
            self.training and self.stochastic_training
            if stochastic is None
            else stochastic
        )
        context, base_logits = self._context(points)
        available = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        selected_features: list[Tensor] = []
        selected_coordinates: list[Tensor] = []
        selected_indices: list[Tensor] = []
        entropies: list[Tensor] = []
        logits_history: list[Tensor] = []
        probabilities_history: list[Tensor] = []

        for _ in range(constellation_size):
            logits = self._conditional_logits(
                context,
                base_logits,
                points,
                selected_features,
                selected_coordinates,
                constellation_size,
            )
            masked_logits = logits.masked_fill(
                ~available, torch.finfo(logits.dtype).min
            )
            sampling_logits = masked_logits
            if stochastic_selection:
                uniform = torch.rand_like(masked_logits).clamp_(1e-6, 1.0 - 1e-6)
                sampling_logits = sampling_logits - torch.log(-torch.log(uniform))
            probabilities = torch.softmax(
                sampling_logits / self.selection_temperature, dim=1
            )
            hard_indices = sampling_logits.argmax(dim=1, keepdim=True)
            hard_weights = torch.zeros_like(probabilities).scatter(1, hard_indices, 1.0)
            weights = hard_weights + probabilities - probabilities.detach()
            selected_features.append(torch.einsum("bn,bnd->bd", weights, context))
            selected_coordinates.append(torch.einsum("bn,bnd->bd", weights, points))
            selected_indices.append(hard_indices.squeeze(1))
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            entropies.append(entropy)
            logits_history.append(masked_logits)
            probabilities_history.append(probabilities)
            available = available.scatter(1, hard_indices, False)

        constellation = torch.stack(selected_coordinates, dim=1)
        constellation = quantize_ste(
            constellation,
            self.bits,
            training=self.training,
            jitter=False,
        )
        if return_trace:
            trace = PointerTrace(
                indices=torch.stack(selected_indices, dim=1),
                entropies=torch.stack(entropies, dim=1),
                logits=tuple(logits_history),
                probabilities=tuple(probabilities_history),
            )
            return constellation, trace
        return constellation

    def beam_search(
        self,
        points: Tensor,
        constellation_size: int,
        *,
        decoder: Decoder,
        target: Tensor,
        num_output_points: int | None = None,
        beam_width: int = 3,
        branch_factor: int = 2,
        score_partial_candidates: bool = False,
    ) -> BeamSearchResult:
        """Search pointer branches and choose by frozen-decoder reconstruction.

        Intermediate candidates use pointer negative log likelihood unless
        ``score_partial_candidates`` is enabled. Completed candidates always use
        decoder Chamfer. The greedy completion is included in the final pool, so
        beam inference cannot return a higher decoder loss than greedy inference.
        """

        self._validate_request(points, constellation_size)
        _validate_points(target, name="target")
        if len(target) != len(points):
            raise ValueError("target and points must have the same batch size")
        if beam_width < 1 or branch_factor < 1:
            raise ValueError("beam_width and branch_factor must be positive")
        if isinstance(decoder, nn.Module) and any(
            parameter.requires_grad for parameter in decoder.parameters()
        ):
            raise ValueError("beam search requires a frozen decoder")

        was_training = self.training
        self.eval()
        with torch.no_grad():
            greedy_output = self(points, constellation_size, return_trace=True)
            assert isinstance(greedy_output, tuple)
            greedy_coordinates, greedy_trace = greedy_output
            greedy_reconstruction = decoder(
                greedy_coordinates, num_output_points=num_output_points
            )
            greedy_losses = _chamfer_per_sample(greedy_reconstruction, target)
            result_indices: list[Tensor] = []
            result_coordinates: list[Tensor] = []
            result_losses: list[Tensor] = []
            evaluation_counts: list[int] = []

            for batch_index in range(len(points)):
                decoder_evaluations = 1
                sample_points = points[batch_index : batch_index + 1]
                sample_target = target[batch_index : batch_index + 1]
                candidates: list[tuple[tuple[int, ...], float, float]] = [
                    ((), 0.0, 0.0)
                ]
                for depth in range(constellation_size):
                    expanded: list[tuple[tuple[int, ...], float, float]] = []
                    for indices, negative_log_likelihood, _ in candidates:
                        selected = torch.tensor(
                            indices,
                            dtype=torch.long,
                            device=points.device,
                        )[None]
                        logits = self.conditional_logits(
                            sample_points, selected, constellation_size
                        )
                        probabilities = torch.softmax(
                            logits / self.selection_temperature, dim=1
                        )
                        branches = min(
                            branch_factor,
                            sample_points.shape[1] - len(indices),
                        )
                        next_indices = logits.topk(branches, dim=1).indices[0]
                        for next_index_tensor in next_indices:
                            next_index = int(next_index_tensor.item())
                            next_tuple = (*indices, next_index)
                            next_nll = negative_log_likelihood - math.log(
                                max(
                                    float(probabilities[0, next_index].item()),
                                    1e-12,
                                )
                            )
                            decoder_loss = math.inf
                            if (
                                score_partial_candidates
                                or depth + 1 == constellation_size
                            ):
                                coordinates = quantize_coordinates(
                                    sample_points[:, list(next_tuple)], self.bits
                                )
                                reconstruction = decoder(
                                    coordinates,
                                    num_output_points=num_output_points,
                                )
                                decoder_evaluations += 1
                                decoder_loss = float(
                                    _chamfer_per_sample(
                                        reconstruction, sample_target
                                    ).item()
                                )
                            priority = (
                                decoder_loss
                                if score_partial_candidates
                                or depth + 1 == constellation_size
                                else next_nll
                            )
                            expanded.append((next_tuple, next_nll, priority))
                    expanded.sort(key=lambda candidate: candidate[2])
                    candidates = expanded[:beam_width]

                greedy_tuple = tuple(
                    int(index) for index in greedy_trace.indices[batch_index].tolist()
                )
                final_tuples = {candidate[0] for candidate in candidates}
                final_tuples.add(greedy_tuple)
                completed = []
                for indices in final_tuples:
                    coordinates = quantize_coordinates(
                        sample_points[:, list(indices)], self.bits
                    )
                    reconstruction = decoder(
                        coordinates, num_output_points=num_output_points
                    )
                    decoder_evaluations += 1
                    loss = _chamfer_per_sample(reconstruction, sample_target)[0]
                    completed.append((loss, indices, coordinates[0]))
                best_loss, best_indices, best_coordinates = min(
                    completed, key=lambda candidate: float(candidate[0].item())
                )
                result_losses.append(best_loss)
                result_indices.append(
                    torch.tensor(best_indices, dtype=torch.long, device=points.device)
                )
                result_coordinates.append(best_coordinates)
                # Count target-scored decoder queries for this cloud.  This is
                # an oracle diagnostic budget, not deployable encoder compute.
                evaluation_counts.append(decoder_evaluations)

        self.train(was_training)
        return BeamSearchResult(
            coordinates=torch.stack(result_coordinates),
            indices=torch.stack(result_indices),
            losses=torch.stack(result_losses),
            greedy_losses=greedy_losses,
            decoder_evaluations=torch.tensor(
                evaluation_counts, dtype=torch.long, device=points.device
            ),
        )
