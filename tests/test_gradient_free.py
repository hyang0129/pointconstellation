"""Focused tests for Experiment 011 decoder-scored search."""

# ruff: noqa: E402

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pointconstellation.gradient_free_experiment import (
    GradientFreeExperimentConfig,
    run_gradient_free_experiment,
)
from pointconstellation.models.gradient_free import (
    adam_ste_search,
    cem_distribution_update,
    coordinate_cem_search,
    mutate_unique_subsets,
    subset_mutation_search,
)
from pointconstellation.quantization import quantization_step, quantize_coordinates


def _quadratic_score(target):
    def score(candidates):
        if candidates.ndim == 3:
            return (candidates - target).square().mean(dim=(1, 2))
        return (candidates - target[:, None]).square().mean(dim=(2, 3))

    return score


def test_cem_distribution_update_selects_elites_and_search_improves() -> None:
    candidates = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]])
    losses = torch.tensor([[4.0, 1.0, 0.0, 9.0]])
    mean, std, elites = cem_distribution_update(
        candidates,
        losses,
        elite_fraction=0.5,
        minimum_std=0.01,
    )
    assert torch.equal(elites.flatten(), torch.tensor([2.0, 1.0]))
    assert mean.item() == pytest.approx(1.5)
    assert std.item() == pytest.approx(0.5)

    initial = torch.zeros(2, 3, 3)
    target = torch.full_like(initial, 0.65)
    result = coordinate_cem_search(
        _quadratic_score(target),
        initial,
        bits=10,
        population_size=64,
        generations=5,
        elite_fraction=0.2,
        initial_std=0.5,
        minimum_std=0.01,
        seed=13,
    )
    assert torch.all(result.losses < result.best_loss_history[0])
    assert torch.all(result.best_loss_history[1:] <= result.best_loss_history[:-1])


def test_unique_subset_mutation_preserves_parent_and_membership() -> None:
    parent = torch.tensor([[0, 2, 4], [1, 3, 5]])
    population = mutate_unique_subsets(
        parent,
        num_points=8,
        population_size=7,
        mutation_swaps=2,
        generator=torch.Generator().manual_seed(17),
    )

    assert population.shape == (2, 7, 3)
    assert torch.equal(population[:, 0], parent)
    assert torch.all((population >= 0) & (population < 8))
    for row in population.flatten(0, 1):
        assert len(torch.unique(row)) == 3


def test_cem_is_deterministic_for_seed() -> None:
    initial = torch.zeros(2, 2, 3)
    target = torch.full_like(initial, -0.35)
    options = {
        "bits": 8,
        "population_size": 8,
        "generations": 3,
        "seed": 29,
    }
    first = coordinate_cem_search(_quadratic_score(target), initial, **options)
    second = coordinate_cem_search(_quadratic_score(target), initial, **options)

    assert torch.equal(first.coordinates, second.coordinates)
    assert torch.equal(first.losses, second.losses)
    assert torch.equal(first.best_loss_history, second.best_loss_history)


def test_subset_search_is_unique_quantized_and_uses_budget() -> None:
    points = torch.linspace(-0.9, 0.9, 10)[:, None].repeat(1, 3)[None]
    initial_indices = torch.tensor([[0, 3, 6, 9]])
    target = torch.zeros(1, 4, 3)
    result = subset_mutation_search(
        _quadratic_score(target),
        points,
        initial_indices,
        bits=8,
        population_size=5,
        generations=3,
        mutation_swaps=1,
        seed=31,
    )
    assert result.indices is not None
    assert len(torch.unique(result.indices[0])) == 4
    quantized_points = quantize_coordinates(points, 8)
    for coordinate in result.coordinates[0]:
        assert torch.any(torch.all(quantized_points[0] == coordinate, dim=1))
    assert result.decoder_evaluations_per_cloud == 16
    assert result.evaluation_counts == (1, 6, 11, 16)


def test_adam_and_cem_budget_pairing_and_exact_quantization() -> None:
    initial = torch.zeros(2, 3, 3)
    target = torch.full_like(initial, 0.2)
    score = _quadratic_score(target)
    cem = coordinate_cem_search(
        score,
        initial,
        bits=8,
        population_size=4,
        generations=2,
        seed=7,
    )
    adam = adam_ste_search(
        score,
        initial,
        bits=8,
        decoder_evaluation_budget=cem.decoder_evaluations_per_cloud,
        learning_rate=0.05,
    )
    step = quantization_step(8)

    assert cem.decoder_evaluations_per_cloud == adam.decoder_evaluations_per_cloud
    assert torch.equal(cem.best_loss_history[0], adam.best_loss_history[0])
    for coordinates in (cem.coordinates, adam.coordinates):
        lattice = (coordinates + 1.0) / step
        assert torch.allclose(lattice, lattice.round(), atol=1e-5)


def test_tiny_run_has_paired_starts_budgets_and_frozen_decoder(tmp_path) -> None:
    config = GradientFreeExperimentConfig(
        num_points=8,
        input_size=8,
        constellation_size=2,
        bits=8,
        train_samples=2,
        validation_samples=1,
        parameter_ood_samples=1,
        batch_size=1,
        decoder_epochs=1,
        feature_width=8,
        num_heads=2,
        population_size=2,
        generations=1,
        perturbation_samples=1,
        output_dir=str(tmp_path),
    )
    result = run_gradient_free_experiment(config, device_name="cpu")
    saved = json.loads((tmp_path / "metrics.json").read_text())

    assert result["decoder_unchanged"]
    assert result["decoder_hash_before_search"] == result["decoder_hash_after_search"]
    assert result["decoder_trainable_parameter_count_during_search"] == 0
    assert (tmp_path / "decoder.pt").exists()
    assert (tmp_path / "search_results.pt").exists()
    for split in ("validation", "parameter_ood"):
        for start_kind in ("fps", "random"):
            comparison = result["evaluation"][split][start_kind]
            assert comparison["paired_initialization"]
            assert comparison["maximum_initial_loss_delta"] <= 1e-6
            assert comparison["matched_search_budget"]
            assert comparison["search_decoder_evaluations_per_cloud"] == 3
            assert set(comparison["methods"]) == {
                "adam_ste",
                "coordinate_cem",
                "subset_mutation",
            }
            for method in comparison["methods"].values():
                assert method["decoder_evaluations_per_cloud"] == 3
                assert method["diagnostics"]["coverage_rmse"] >= 0
                assert method["diagnostics"]["surface_proxy_rmse"] >= 0
                assert method["diagnostics"]["perturbation_sensitivity"] >= 0
    assert saved["score_contract"] == "decoder reconstruction Chamfer only"
    assert saved["message_contract"]["exact_final_quantization"]
