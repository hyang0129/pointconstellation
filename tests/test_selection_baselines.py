# ruff: noqa: E402

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.selection_baselines import SELECTION_METHODS


def _sphere(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(count, 3, generator=generator)
    return points / points.norm(dim=1, keepdim=True)


def _scorer(candidate: torch.Tensor) -> float:
    return float((candidate**2).mean().item())


@pytest.mark.parametrize("method", sorted(SELECTION_METHODS))
def test_selection_methods_are_deterministic_exact_size_and_on_lattice(
    method: str,
) -> None:
    points = _sphere(64, seed=11)
    selected = SELECTION_METHODS[method](points, 8, 12, 29, _scorer)
    repeated = SELECTION_METHODS[method](points, 8, 12, 29, _scorer)
    lattice = (selected + 1.0) * 0.5 * ((1 << 12) - 1)

    assert selected.shape == (8, 3)
    assert torch.equal(selected, repeated)
    assert torch.allclose(lattice, lattice.round(), atol=5e-4, rtol=0.0)


def test_poisson_disk_returns_requested_sphere_subset_size() -> None:
    selected = SELECTION_METHODS["poisson_disk"](_sphere(128, seed=7), 12, 10, 41, None)

    assert selected.shape == (12, 3)


@pytest.mark.parametrize("method", ("kmeans", "kmeans_weighted"))
def test_kmeans_centroids_stay_inside_source_bbox(method: str) -> None:
    generator = torch.Generator().manual_seed(101)
    random_points = 1.6 * torch.rand(62, 3, generator=generator) - 0.8
    points = torch.cat((random_points, -torch.ones(1, 3), torch.ones(1, 3)))
    selected = SELECTION_METHODS[method](points, 8, 12, 17, None)

    assert torch.all(selected >= points.amin(dim=0))
    assert torch.all(selected <= points.amax(dim=0))
