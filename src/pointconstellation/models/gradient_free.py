"""Gradient-free and gradient-based search over literal constellations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pointconstellation.quantization import quantize_coordinates, quantize_ste

ScoreFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class SearchResult:
    """Best candidates and accounting from a decoder-scored search."""

    coordinates: Tensor
    losses: Tensor
    evaluation_counts: tuple[int, ...]
    best_loss_history: Tensor
    decoder_evaluations_per_cloud: int
    indices: Tensor | None = None


def _validate_coordinates(coordinates: Tensor) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (batch, K, 3)")
    if coordinates.shape[1] < 1:
        raise ValueError("constellations cannot be empty")
    if not coordinates.is_floating_point():
        raise ValueError("coordinates must be floating point")


def _cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _randn_like_shape(reference: Tensor, shape: tuple[int, ...], generator) -> Tensor:
    return torch.randn(shape, dtype=reference.dtype, generator=generator).to(
        reference.device
    )


def cem_distribution_update(
    candidates: Tensor,
    losses: Tensor,
    *,
    elite_fraction: float,
    minimum_std: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit an independent Gaussian to each batch item's lowest-loss elites."""

    if candidates.ndim < 3:
        raise ValueError("candidates must have shape (batch, population, ...)")
    if losses.shape != candidates.shape[:2]:
        raise ValueError("losses must have shape (batch, population)")
    if not 0 < elite_fraction <= 1:
        raise ValueError("elite_fraction must be in (0, 1]")
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    elite_count = max(1, math.ceil(candidates.shape[1] * elite_fraction))
    elite_indices = losses.topk(elite_count, largest=False, dim=1).indices
    gather_shape = elite_indices.shape + (1,) * (candidates.ndim - 2)
    expanded = elite_indices.reshape(gather_shape).expand(-1, -1, *candidates.shape[2:])
    elites = candidates.gather(1, expanded)
    mean = elites.mean(dim=1)
    standard_deviation = elites.std(dim=1, unbiased=False).clamp_min(minimum_std)
    return mean, standard_deviation, elites


def coordinate_cem_search(
    score: ScoreFunction,
    initial_coordinates: Tensor,
    *,
    bits: int,
    population_size: int,
    generations: int,
    elite_fraction: float = 0.25,
    initial_std: float = 0.1,
    minimum_std: float = 0.005,
    seed: int = 7,
) -> SearchResult:
    """Search unrestricted coordinate space with the cross-entropy method."""

    _validate_coordinates(initial_coordinates)
    if population_size < 2 or generations < 1:
        raise ValueError(
            "population_size must be at least two and generations positive"
        )
    if initial_std <= 0:
        raise ValueError("initial_std must be positive")

    generator = _cpu_generator(seed)
    mean = initial_coordinates.detach().clamp(-1.0, 1.0)
    standard_deviation = torch.full_like(mean, initial_std)
    best = quantize_coordinates(mean, bits)
    with torch.no_grad():
        best_losses = score(best)
    if best_losses.shape != (len(best),):
        raise ValueError("score must return one loss per batch item")
    history = [best_losses.detach().clone()]
    evaluation_counts = [1]
    rows = torch.arange(len(best), device=best.device)

    for generation in range(1, generations + 1):
        noise = _randn_like_shape(
            mean,
            (len(mean), population_size, *mean.shape[1:]),
            generator,
        )
        candidates = mean[:, None] + standard_deviation[:, None] * noise
        candidates = quantize_coordinates(candidates.clamp(-1.0, 1.0), bits)
        candidates[:, 0] = best
        with torch.no_grad():
            losses = score(candidates)
        if losses.shape != (len(best), population_size):
            raise ValueError("score must preserve batch and population dimensions")
        mean, standard_deviation, _ = cem_distribution_update(
            candidates,
            losses,
            elite_fraction=elite_fraction,
            minimum_std=minimum_std,
        )
        generation_losses, generation_indices = losses.min(dim=1)
        improved = generation_losses < best_losses
        generation_best = candidates[rows, generation_indices]
        best = torch.where(improved[:, None, None], generation_best, best)
        best_losses = torch.minimum(best_losses, generation_losses)
        history.append(best_losses.detach().clone())
        evaluation_counts.append(1 + generation * population_size)

    return SearchResult(
        coordinates=quantize_coordinates(best.detach(), bits),
        losses=best_losses.detach(),
        evaluation_counts=tuple(evaluation_counts),
        best_loss_history=torch.stack(history),
        decoder_evaluations_per_cloud=evaluation_counts[-1],
    )


def mutate_unique_subsets(
    indices: Tensor,
    *,
    num_points: int,
    population_size: int,
    mutation_swaps: int = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Create a population of unique subsets, retaining the parent at index 0."""

    if indices.ndim != 2:
        raise ValueError("indices must have shape (batch, K)")
    if indices.shape[1] > num_points:
        raise ValueError("subset cannot be larger than num_points")
    if population_size < 2 or mutation_swaps < 1:
        raise ValueError("population_size must be at least two and swaps positive")
    if torch.any(indices < 0) or torch.any(indices >= num_points):
        raise ValueError("indices are outside the input point range")
    if any(len(torch.unique(row)) != len(row) for row in indices):
        raise ValueError("every parent subset must contain unique indices")

    cpu_indices = indices.detach().cpu()
    population = cpu_indices[:, None].expand(-1, population_size, -1).clone()
    random = generator or _cpu_generator(0)
    if indices.shape[1] == num_points:
        return population.to(indices.device)

    for batch_index in range(len(cpu_indices)):
        for candidate_index in range(1, population_size):
            candidate = population[batch_index, candidate_index]
            for _ in range(min(mutation_swaps, indices.shape[1])):
                slot = int(
                    torch.randint(indices.shape[1], (1,), generator=random).item()
                )
                present = torch.zeros(num_points, dtype=torch.bool)
                present[candidate] = True
                available = torch.arange(num_points)[~present]
                replacement = available[
                    torch.randint(len(available), (1,), generator=random)
                ]
                candidate[slot] = replacement
    return population.to(indices.device)


def subset_mutation_search(
    score: ScoreFunction,
    points: Tensor,
    initial_indices: Tensor,
    *,
    bits: int,
    population_size: int,
    generations: int,
    mutation_swaps: int = 1,
    seed: int = 7,
) -> SearchResult:
    """Search strict input subsets with decoder-scored swap mutations."""

    _validate_coordinates(points)
    if initial_indices.ndim != 2:
        raise ValueError("initial_indices must have shape (batch, K)")
    if len(initial_indices) != len(points):
        raise ValueError("initial_indices batch size must match points")
    if generations < 1:
        raise ValueError("generations must be positive")

    quantized_points = quantize_coordinates(points, bits)
    best_indices = initial_indices.detach().clone()
    best = quantized_points.gather(1, best_indices[:, :, None].expand(-1, -1, 3))
    with torch.no_grad():
        best_losses = score(best)
    history = [best_losses.detach().clone()]
    evaluation_counts = [1]
    generator = _cpu_generator(seed)
    batch_rows = torch.arange(len(points), device=points.device)

    for generation in range(1, generations + 1):
        candidate_indices = mutate_unique_subsets(
            best_indices,
            num_points=points.shape[1],
            population_size=population_size,
            mutation_swaps=mutation_swaps,
            generator=generator,
        )
        expanded_points = quantized_points[:, None].expand(-1, population_size, -1, -1)
        candidates = expanded_points.gather(
            2, candidate_indices[:, :, :, None].expand(-1, -1, -1, 3)
        )
        with torch.no_grad():
            losses = score(candidates)
        generation_losses, generation_indices = losses.min(dim=1)
        improved = generation_losses < best_losses
        generation_best_indices = candidate_indices[batch_rows, generation_indices]
        generation_best = candidates[batch_rows, generation_indices]
        best_indices = torch.where(
            improved[:, None], generation_best_indices, best_indices
        )
        best = torch.where(improved[:, None, None], generation_best, best)
        best_losses = torch.minimum(best_losses, generation_losses)
        history.append(best_losses.detach().clone())
        evaluation_counts.append(1 + generation * population_size)

    return SearchResult(
        coordinates=quantize_coordinates(best.detach(), bits),
        losses=best_losses.detach(),
        evaluation_counts=tuple(evaluation_counts),
        best_loss_history=torch.stack(history),
        decoder_evaluations_per_cloud=evaluation_counts[-1],
        indices=best_indices.detach(),
    )


def adam_ste_search(
    score: ScoreFunction,
    initial_coordinates: Tensor,
    *,
    bits: int,
    decoder_evaluation_budget: int,
    learning_rate: float = 0.03,
) -> SearchResult:
    """Optimize coordinates with Adam/STE under an explicit decoder budget."""

    _validate_coordinates(initial_coordinates)
    if decoder_evaluation_budget < 1:
        raise ValueError("decoder_evaluation_budget must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    raw = nn.Parameter(initial_coordinates.detach().clone())
    optimizer = torch.optim.Adam((raw,), lr=learning_rate)
    best = quantize_coordinates(raw.detach(), bits)
    best_losses = torch.full((len(raw),), torch.inf, dtype=raw.dtype, device=raw.device)
    history = []

    for evaluation in range(decoder_evaluation_budget):
        optimizer.zero_grad(set_to_none=True)
        coordinates = quantize_ste(
            raw.clamp(-1.0, 1.0), bits, training=True, jitter=False
        )
        losses = score(coordinates)
        if losses.shape != (len(raw),):
            raise ValueError("score must return one loss per batch item")
        with torch.no_grad():
            improved = losses < best_losses
            best = torch.where(improved[:, None, None], coordinates.detach(), best)
            best_losses = torch.minimum(best_losses, losses.detach())
            history.append(best_losses.clone())
        if evaluation + 1 < decoder_evaluation_budget:
            losses.mean().backward()
            optimizer.step()
            with torch.no_grad():
                raw.clamp_(-1.0, 1.0)

    return SearchResult(
        coordinates=quantize_coordinates(best.detach(), bits),
        losses=best_losses.detach(),
        evaluation_counts=tuple(range(1, decoder_evaluation_budget + 1)),
        best_loss_history=torch.stack(history),
        decoder_evaluations_per_cloud=decoder_evaluation_budget,
    )
