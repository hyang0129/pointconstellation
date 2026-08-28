from __future__ import annotations

import numpy as np

from pointconstellation.irregularity import (
    decoder_residual_score,
    deterministic_random_scores,
    local_geometry_scores,
    select_spaced_indices,
    stratification_bins,
    stratified_error,
)


def test_irregularity_scores_are_permutation_equivariant_and_deterministic() -> None:
    rng = np.random.default_rng(17)
    points = rng.uniform(-1.0, 1.0, size=(48, 3))
    reconstruction = rng.uniform(-1.0, 1.0, size=(31, 3))
    permutation = rng.permutation(len(points))
    inverse = np.argsort(permutation)
    reconstruction_permutation = rng.permutation(len(reconstruction))

    first = local_geometry_scores(points, neighbors=7, chunk_size=11)
    repeated = local_geometry_scores(points, neighbors=7, chunk_size=11)
    permuted = local_geometry_scores(points[permutation], neighbors=7, chunk_size=13)

    for name in ("curvature", "density_deviation", "boundary", "thin_structure"):
        assert np.allclose(getattr(first, name), getattr(repeated, name), atol=0.0)
        assert np.allclose(
            getattr(first, name), getattr(permuted, name)[inverse], atol=1e-12
        )
    residual = decoder_residual_score(points, reconstruction, chunk_size=9)
    residual_permuted = decoder_residual_score(
        points[permutation], reconstruction[reconstruction_permutation], chunk_size=8
    )
    assert np.allclose(residual, residual_permuted[inverse], atol=1e-12)
    random_scores = deterministic_random_scores(points, seed=101)
    permuted_random = deterministic_random_scores(points[permutation], seed=101)
    assert np.array_equal(random_scores, permuted_random[inverse])


def test_spaced_selection_is_order_independent_and_strata_cover_cloud() -> None:
    rng = np.random.default_rng(29)
    points = rng.uniform(-1.0, 1.0, size=(53, 3))
    scores = deterministic_random_scores(points, seed=211)
    permutation = rng.permutation(len(points))
    selected = select_spaced_indices(points, scores, 9, minimum_spacing=0.1)
    selected_permuted = select_spaced_indices(
        points[permutation], scores[permutation], 9, minimum_spacing=0.1
    )
    first_points = sorted(map(tuple, points[selected].tolist()))
    second_points = sorted(map(tuple, points[permutation][selected_permuted].tolist()))

    assert first_points == second_points
    distances = np.linalg.norm(
        points[selected, None] - points[selected][None, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 0.1

    assignments = stratification_bins(scores, points=points)
    summary = stratified_error(np.linspace(0.0, 1.0, len(points)), assignments)
    assert sum(row["count"] for row in summary) == len(points)
    assert set(assignments) == {0, 1, 2, 3, 4}
