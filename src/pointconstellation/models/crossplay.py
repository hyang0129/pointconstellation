"""Decoder-population utilities for constellation cross-play experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pointconstellation.losses import chamfer_squared
from pointconstellation.models.bottleneck import VariableConstellationDecoder
from pointconstellation.models.refiner import CompetitiveConstellationRefiner


class DecoderPopulation(nn.Module):
    """A population of independently initialized coordinate-only decoders."""

    def __init__(self, decoders: Sequence[VariableConstellationDecoder]) -> None:
        super().__init__()
        if len(decoders) < 2:
            raise ValueError("a cross-play population requires at least two decoders")
        if len({id(decoder) for decoder in decoders}) != len(decoders):
            raise ValueError("population entries must be distinct decoder instances")
        self.decoders = nn.ModuleList(decoders)

    def __len__(self) -> int:
        return len(self.decoders)

    def forward_all(
        self, constellation: Tensor, *, num_output_points: int
    ) -> list[Tensor]:
        """Decode one coordinate message with every population member."""

        return [
            decoder(constellation, num_output_points=num_output_points)
            for decoder in self.decoders
        ]


def make_decoder_population(
    population_size: int,
    *,
    max_output_points: int,
    max_constellation_size: int,
    feature_width: int,
    num_heads: int,
    num_layers: int,
    seed: int,
) -> DecoderPopulation:
    """Construct reproducibly independent decoder initializations.

    ``fork_rng`` prevents the population construction from changing the caller's
    CPU random stream. Every member is seeded separately and owns its parameters.
    """

    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    decoders: list[VariableConstellationDecoder] = []
    with torch.random.fork_rng(devices=[]):
        for index in range(population_size):
            torch.manual_seed(seed + index)
            decoders.append(
                VariableConstellationDecoder(
                    max_output_points,
                    max_constellation_size,
                    feature_width=feature_width,
                    num_heads=num_heads,
                    num_layers=num_layers,
                )
            )
    return DecoderPopulation(decoders)


def make_crossplay_refiner(
    max_constellation_size: int,
    *,
    bits: int,
    feature_width: int,
    num_heads: int,
    recurrent_steps: int,
    responsibility_temperature: float,
    maximum_update: float,
) -> CompetitiveConstellationRefiner:
    """Build the common encoder architecture used by both H11 conditions."""

    return CompetitiveConstellationRefiner(
        max_constellation_size,
        bits=bits,
        feature_width=feature_width,
        num_heads=num_heads,
        recurrent_steps=recurrent_steps,
        responsibility_temperature=responsibility_temperature,
        maximum_update=maximum_update,
        # Decoder-gradient feedback would expose a different decoder-conditioned
        # input to each arm. H11 instead changes only the reconstruction objective.
        use_decoder_gradient=False,
    )


def aggregate_decoder_losses(losses: Tensor, *, worst_weight: float = 0.0) -> Tensor:
    """Return mean population loss plus an optional worst-partner penalty."""

    if losses.ndim != 1 or losses.numel() < 1:
        raise ValueError("losses must be a nonempty one-dimensional tensor")
    if worst_weight < 0:
        raise ValueError("worst_weight cannot be negative")
    return losses.mean() + worst_weight * losses.max()


def reconstruction_loss_matrix(
    messages: Sequence[Tensor],
    population: DecoderPopulation,
    target: Tensor,
    *,
    num_output_points: int,
) -> Tensor:
    """Return encoder-message x decoder scalar Chamfer losses.

    Rows correspond to message-producing encoders and columns to independent
    decoders. This is the complete cross-play matrix for one batch.
    """

    if not messages:
        raise ValueError("at least one encoder message is required")
    rows = []
    for message in messages:
        rows.append(
            torch.stack(
                [
                    chamfer_squared(reconstruction, target)
                    for reconstruction in population.forward_all(
                        message, num_output_points=num_output_points
                    )
                ]
            )
        )
    return torch.stack(rows)
