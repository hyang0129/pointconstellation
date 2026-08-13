"""Mass-aware point-set losses for constellation experiments.

The functions in this module deliberately avoid optional optimal-transport
dependencies.  The balanced transport plan is computed in the log domain, so
it remains differentiable while being stable for the small entropic
regularization values used by the experiments.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from pointconstellation.losses import pairwise_squared
from pointconstellation.models.refiner import CompetitiveConstellationRefiner


def _validate_point_pair(first: Tensor, second: Tensor) -> None:
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError("first must have shape (batch, M, 3)")
    if second.ndim != 3 or second.shape[-1] != 3:
        raise ValueError("second must have shape (batch, N, 3)")
    if first.shape[0] != second.shape[0]:
        raise ValueError("point sets must have the same batch size")
    if first.shape[1] < 1 or second.shape[1] < 1:
        raise ValueError("point sets cannot be empty")
    if not first.is_floating_point() or not second.is_floating_point():
        raise ValueError("point sets must use floating-point coordinates")


def sinkhorn_transport_plan(
    first: Tensor,
    second: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 40,
) -> Tensor:
    """Return a balanced entropic transport plan with uniform marginals.

    The returned tensor has shape ``(batch, M, N)``.  Its rows sum to ``1/M``
    and columns to ``1/N`` up to the finite Sinkhorn iteration tolerance.  No
    transport plan is detached, so gradients propagate through both the cost
    matrix and the normalization iterations.
    """

    _validate_point_pair(first, second)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    cost = pairwise_squared(first, second)
    batch_size, first_count, second_count = cost.shape
    log_kernel = -cost / epsilon
    log_first_mass = cost.new_full((batch_size, first_count), -math.log(first_count))
    log_second_mass = cost.new_full((batch_size, second_count), -math.log(second_count))
    log_u = torch.zeros_like(log_first_mass)
    log_v = torch.zeros_like(log_second_mass)
    for _ in range(iterations):
        log_u = log_first_mass - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2)
        log_v = log_second_mass - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1)
    return torch.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])


def balanced_transport_squared(
    first: Tensor,
    second: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 40,
) -> Tensor:
    """Return the mean squared cost of balanced entropic transport."""

    plan = sinkhorn_transport_plan(
        first, second, epsilon=epsilon, iterations=iterations
    )
    return (plan * pairwise_squared(first, second)).sum(dim=(1, 2)).mean()


def balanced_anchor_responsibilities(
    anchors: Tensor,
    points: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 40,
) -> Tensor:
    """Assign each point to anchors while enforcing equal anchor mass.

    Responsibilities have shape ``(batch, K, N)``.  Each point distributes one
    unit of responsibility across anchors and every anchor receives ``N/K``
    units.  This scaling is more convenient for slot diagnostics than a joint
    probability transport plan.
    """

    plan = sinkhorn_transport_plan(
        anchors, points, epsilon=epsilon, iterations=iterations
    )
    return plan * points.shape[1]


class BalancedResponsibilityRefiner(CompetitiveConstellationRefiner):
    """Competitive refiner whose internal point allocation is balanced OT.

    It has exactly the same trainable parameter structure as the base refiner,
    allowing bitwise-matched initialization.  The only intervention is the
    K-by-N responsibility matrix used to aggregate point evidence.
    """

    def __init__(
        self,
        *args: object,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 40,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive")
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

    def competitive_responsibilities(
        self,
        slot_features: Tensor,
        coordinates: Tensor,
        point_features: Tensor,
        points: Tensor,
    ) -> Tensor:
        del slot_features, point_features
        return balanced_anchor_responsibilities(
            coordinates,
            points,
            epsilon=self.sinkhorn_epsilon,
            iterations=self.sinkhorn_iterations,
        )


def balanced_anchor_transport_loss(
    anchors: Tensor,
    points: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 40,
) -> Tensor:
    """Encourage anchors to cover equal-mass regions of a point set."""

    return balanced_transport_squared(
        anchors, points, epsilon=epsilon, iterations=iterations
    )


def density_aware_chamfer_squared(
    first: Tensor,
    second: Tensor,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """A smooth density-aware Chamfer-style discrepancy.

    Soft nearest-neighbour assignments estimate how many points compete for
    each match.  Dividing match affinity by that occupancy makes duplicate or
    collapsed predictions costly even when they lie exactly on an observed
    target point.  The symmetric construction is permutation invariant and
    differentiable with respect to both point sets.
    """

    _validate_point_pair(first, second)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    distances = pairwise_squared(first, second)
    affinity = torch.exp(-distances / temperature)

    first_to_second = torch.softmax(-distances / temperature, dim=2)
    second_occupancy = first_to_second.sum(dim=1, keepdim=True)
    expected_second_occupancy = first.shape[1] / second.shape[1]
    second_density_discount = 1.0 / (
        1.0 + (second_occupancy / expected_second_occupancy - 1.0).square()
    )
    first_match = (first_to_second * affinity * second_density_discount).sum(dim=2)

    second_to_first = torch.softmax(-distances / temperature, dim=1)
    first_occupancy = second_to_first.sum(dim=2, keepdim=True)
    expected_first_occupancy = second.shape[1] / first.shape[1]
    first_density_discount = 1.0 / (
        1.0 + (first_occupancy / expected_first_occupancy - 1.0).square()
    )
    second_match = (second_to_first * affinity * first_density_discount).sum(dim=1)

    return 0.5 * ((1.0 - first_match).mean() + (1.0 - second_match).mean())


def soft_anchor_responsibilities(
    anchors: Tensor,
    points: Tensor,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Return unconstrained point-to-anchor probabilities for diagnostics."""

    _validate_point_pair(anchors, points)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    distances = pairwise_squared(anchors, points)
    return torch.softmax(-distances / temperature, dim=1)


def anchor_mass_imbalance(
    anchors: Tensor,
    points: Tensor,
    *,
    temperature: float = 0.05,
) -> Tensor:
    """Return relative RMS deviation from equal unconstrained anchor mass.

    Zero means that each anchor explains exactly ``1/K`` of the point mass.  A
    value of one means the RMS deviation equals the desired per-anchor mass.
    The metric is averaged over the batch.
    """

    responsibilities = soft_anchor_responsibilities(
        anchors, points, temperature=temperature
    )
    mass_fraction = responsibilities.mean(dim=2)
    expected = 1.0 / anchors.shape[1]
    relative_squared = ((mass_fraction - expected) / expected).square()
    return relative_squared.mean(dim=1).sqrt().mean()
