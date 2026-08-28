"""Deterministic non-learned selectors for fixed-rate constellations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from pointconstellation.quantization import quantize_ste

SelectionScorer = Callable[[Tensor], float | Tensor]
SelectionMethod = Callable[[Tensor, int, int, int, SelectionScorer | None], Tensor]

KMEANS_MAX_ITERATIONS = 50
KMEANS_DENSITY_NEIGHBORS = 8
POISSON_BISECTION_STEPS = 48


def _validated_points(points: Tensor, constellation_size: int) -> Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not torch.isfinite(points).all():
        raise ValueError("points must be finite")
    if not 1 <= constellation_size <= len(points):
        raise ValueError("constellation_size must be between one and N")
    return points.detach()


def _quantize(points: Tensor, bits: int) -> Tensor:
    return quantize_ste(points, bits, training=False, jitter=False)


def _quantize_centroids(centers: Tensor, source: Tensor, bits: int) -> Tensor:
    """Quantize centroids without crossing a source bbox face when feasible."""

    levels = (1 << bits) - 1
    indices = torch.round((centers + 1.0) * 0.5 * levels).clamp(0, levels)
    lower = torch.ceil((source.amin(dim=0) + 1.0) * 0.5 * levels).clamp(0, levels)
    upper = torch.floor((source.amax(dim=0) + 1.0) * 0.5 * levels).clamp(0, levels)
    feasible = lower <= upper
    constrained = torch.maximum(torch.minimum(indices, upper), lower)
    indices = torch.where(feasible, constrained, indices)
    return _quantize(indices * (2.0 / levels) - 1.0, bits)


def _squared_distances(first: Tensor, second: Tensor) -> Tensor:
    first_norm = (first**2).sum(dim=1, keepdim=True)
    second_norm = (second**2).sum(dim=1).unsqueeze(0)
    return (first_norm + second_norm - 2.0 * first @ second.T).clamp_min(0.0)


def _choice(
    rng: np.random.Generator, probabilities: Tensor, unavailable: set[int]
) -> int:
    values = probabilities.detach().double().cpu().numpy()
    if unavailable:
        values[np.fromiter(unavailable, dtype=np.int64)] = 0.0
    total = float(values.sum())
    if np.isfinite(total) and total > 0.0:
        return int(rng.choice(len(values), p=values / total))
    available = np.asarray(
        [index for index in range(len(values)) if index not in unavailable],
        dtype=np.int64,
    )
    return int(rng.choice(available))


def _kmeans_plus_plus(
    points: Tensor,
    constellation_size: int,
    weights: Tensor,
    rng: np.random.Generator,
) -> Tensor:
    selected: list[int] = []
    first = _choice(rng, weights, set())
    selected.append(first)
    nearest = _squared_distances(points, points[first : first + 1]).squeeze(1)
    while len(selected) < constellation_size:
        index = _choice(rng, weights * nearest, set(selected))
        selected.append(index)
        distance = _squared_distances(points, points[index : index + 1]).squeeze(1)
        nearest = torch.minimum(nearest, distance)
    indices = torch.as_tensor(selected, dtype=torch.long, device=points.device)
    return points[indices].clone()


def _lloyd_kmeans(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    weights: Tensor,
) -> Tensor:
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus(points, constellation_size, weights, rng)
    previous_assignments: Tensor | None = None
    for _ in range(KMEANS_MAX_ITERATIONS):
        distances = _squared_distances(points, centers)
        assignments = distances.argmin(dim=1)
        updated = centers.clone()
        for cluster in range(constellation_size):
            members = assignments == cluster
            if members.any():
                member_weights = weights[members]
                updated[cluster] = (points[members] * member_weights[:, None]).sum(
                    dim=0
                ) / member_weights.sum()
        converged = previous_assignments is not None and torch.equal(
            assignments, previous_assignments
        )
        centers = updated
        previous_assignments = assignments
        if converged:
            break
    return _quantize_centroids(centers, points, bits)


def kmeans(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return fixed-seed k-means++/Lloyd free-coordinate centroids."""

    del scorer
    points = _validated_points(points, constellation_size)
    device = points.device
    points = points.cpu()
    weights = torch.ones(len(points), dtype=points.dtype)
    return _lloyd_kmeans(points, constellation_size, bits, seed, weights).to(device)


def _inverse_density_weights(points: Tensor) -> Tensor:
    if len(points) == 1:
        return torch.ones(1, dtype=points.dtype, device=points.device)
    neighbor_count = min(KMEANS_DENSITY_NEIGHBORS, len(points) - 1)
    distances = _squared_distances(points, points)
    distances.fill_diagonal_(torch.inf)
    radius = distances.kthvalue(neighbor_count, dim=1).values.clamp_min(0.0).sqrt()
    weights = radius.pow(3)
    positive = weights > 0
    if not positive.any():
        return torch.ones_like(weights)
    floor = weights[positive].amin() * 1e-6
    weights = weights.clamp_min(floor)
    return weights / weights.mean()


def kmeans_weighted(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return k-means centroids weighted by an inverse-density proxy."""

    del scorer
    points = _validated_points(points, constellation_size)
    device = points.device
    points = points.cpu()
    weights = _inverse_density_weights(points)
    return _lloyd_kmeans(points, constellation_size, bits, seed, weights).to(device)


def _poisson_greedy(
    points: np.ndarray,
    order: np.ndarray,
    radius_squared: float,
    limit: int,
) -> list[int]:
    selected: list[int] = []
    for index in order:
        index = int(index)
        if not selected:
            selected.append(index)
        else:
            distances = ((points[np.asarray(selected)] - points[index]) ** 2).sum(
                axis=1
            )
            if np.all(distances >= radius_squared):
                selected.append(index)
        if len(selected) == limit:
            break
    return selected


def poisson_disk(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return a seeded greedy subset at the largest bisected feasible radius."""

    del scorer
    points = _validated_points(points, constellation_size)
    rng = np.random.default_rng(seed)
    values = points.detach().cpu().numpy()
    order = rng.permutation(len(points))
    lower = 0.0
    diagonal = np.linalg.norm(values.max(axis=0) - values.min(axis=0))
    upper = float(diagonal) * (1.0 + 1e-6) + 1e-12
    best = _poisson_greedy(values, order, lower, constellation_size)
    for _ in range(POISSON_BISECTION_STEPS):
        radius = 0.5 * (lower + upper)
        selected = _poisson_greedy(values, order, radius * radius, constellation_size)
        if len(selected) >= constellation_size:
            lower = radius
            best = selected
        else:
            upper = radius
    indices = torch.as_tensor(
        best[:constellation_size], dtype=torch.long, device=points.device
    )
    return _quantize(points[indices], bits)


def _fps_from_start(points: Tensor, constellation_size: int, start: int) -> Tensor:
    selected = torch.empty(constellation_size, dtype=torch.long, device=points.device)
    minimum = torch.full(
        (len(points),), torch.inf, dtype=points.dtype, device=points.device
    )
    farthest = start
    for index in range(constellation_size):
        selected[index] = farthest
        distance = ((points - points[farthest]) ** 2).sum(dim=1)
        minimum = torch.minimum(minimum, distance)
        farthest = int(minimum.argmax().item())
    return points[selected]


def fps(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return the existing deterministic farthest-point-sampling baseline."""

    del seed, scorer
    points = _validated_points(points, constellation_size)
    start = int(((points - points.mean(dim=0)) ** 2).sum(dim=1).argmax().item())
    return _quantize(_fps_from_start(points, constellation_size, start), bits)


def _score(candidate: Tensor, scorer: SelectionScorer) -> float:
    value = scorer(candidate)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("selection scorer must return a scalar")
        result = float(value.detach().item())
    else:
        result = float(value)
    if not np.isfinite(result):
        raise ValueError("selection scorer must return a finite value")
    return result


def _best_candidate(candidates: list[Tensor], scorer: SelectionScorer | None) -> Tensor:
    if scorer is None:
        if len(candidates) != 1:
            raise ValueError("best-of-N selection requires a scorer when N > 1")
        return candidates[0]
    scores = [_score(candidate, scorer) for candidate in candidates]
    return candidates[int(np.argmin(np.asarray(scores)))]


def _fps_random_start(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
    *,
    trials: int,
) -> Tensor:
    points = _validated_points(points, constellation_size)
    rng = np.random.default_rng(seed)
    if trials <= len(points):
        starts = rng.permutation(len(points))[:trials]
    else:
        starts = rng.integers(0, len(points), size=trials)
    candidates = [
        _quantize(_fps_from_start(points, constellation_size, int(start)), bits)
        for start in starts
    ]
    return _best_candidate(candidates, scorer)


def fps_random_start(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return one FPS subset from a seeded random start."""

    return _fps_random_start(points, constellation_size, bits, seed, scorer, trials=1)


def fps_random_start_best_of_8(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return the best of eight random-start FPS subsets by decoder score."""

    return _fps_random_start(points, constellation_size, bits, seed, scorer, trials=8)


def _random_best_of_n(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
    *,
    trials: int,
) -> Tensor:
    points = _validated_points(points, constellation_size)
    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(trials):
        indices = torch.as_tensor(
            rng.choice(len(points), size=constellation_size, replace=False),
            dtype=torch.long,
            device=points.device,
        )
        candidates.append(_quantize(points[indices], bits))
    return _best_candidate(candidates, scorer)


def random_best_of_1(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return one seeded random strict subset."""

    return _random_best_of_n(points, constellation_size, bits, seed, scorer, trials=1)


def random_best_of_16(
    points: Tensor,
    constellation_size: int,
    bits: int,
    seed: int,
    scorer: SelectionScorer | None,
) -> Tensor:
    """Return the best of sixteen random subsets by decoder score."""

    return _random_best_of_n(points, constellation_size, bits, seed, scorer, trials=16)


SELECTION_METHODS: dict[str, SelectionMethod] = {
    "fps": fps,
    "kmeans": kmeans,
    "kmeans_weighted": kmeans_weighted,
    "poisson_disk": poisson_disk,
    "fps_random_start": fps_random_start,
    "fps_random_start_best_of_8": fps_random_start_best_of_8,
    "random_best_of_1": random_best_of_1,
    "random_best_of_16": random_best_of_16,
}

SELECTION_REPRESENTATIONS = {
    "fps": "strict-subset",
    "kmeans": "free-coordinate",
    "kmeans_weighted": "free-coordinate",
    "poisson_disk": "strict-subset",
    "fps_random_start": "strict-subset",
    "fps_random_start_best_of_8": "strict-subset",
    "random_best_of_1": "strict-subset",
    "random_best_of_16": "strict-subset",
}

SELECTION_TRIALS = {
    "fps": 1,
    "kmeans": 1,
    "kmeans_weighted": 1,
    "poisson_disk": 1,
    "fps_random_start": 1,
    "fps_random_start_best_of_8": 8,
    "random_best_of_1": 1,
    "random_best_of_16": 16,
}
