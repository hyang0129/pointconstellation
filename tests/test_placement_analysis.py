"""Focused tests for Experiment 034 placement metrics and figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.bitstream import decode_constellation, encode_constellation
from pointconstellation.placement_analysis import (
    METHODS,
    PlacementAnalysisConfig,
    _load_rows,
    category_consistency,
    generate_category_maps_svg,
    generate_qualitative_svg,
    mesh_proxy_values,
    part_hit_fractions,
    placement_entropy,
    summarize_placement_rows,
)


def _structured_constellations() -> tuple[np.ndarray, list[str], list[str]]:
    template = np.asarray(
        [
            [-0.04, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, -0.04, 0.00],
            [0.00, 0.04, 0.00],
        ]
    )
    constellations = []
    categories = []
    instance_ids = []
    for category_index, center in enumerate((-0.65, 0.65)):
        for instance_index, jitter in enumerate((-0.005, 0.005)):
            constellations.append(template + [center + jitter, 0.0, 0.0])
            categories.append(f"category_{category_index}")
            instance_ids.append(f"instance_{category_index}_{instance_index}")
    return np.asarray(constellations), categories, instance_ids


def test_category_consistency_known_structure_and_permutation_invariance() -> None:
    constellations, categories, instance_ids = _structured_constellations()

    result = category_consistency(constellations, categories, instance_ids)
    permuted = category_consistency(
        constellations[:, [2, 0, 3, 1]], categories, instance_ids
    )

    assert result["within_category_mean"] < 0.02
    assert result["across_category_mean"] > 1.0
    assert result["within_to_across_ratio"] < 0.02
    assert result["within_to_across_ratio"] == pytest.approx(
        permuted["within_to_across_ratio"]
    )


def test_entropy_reports_known_equal_voxel_occupancy() -> None:
    points = np.asarray(
        [
            [-0.75, -0.75, -0.75],
            [-0.75, -0.75, 0.75],
            [0.75, 0.75, -0.75],
            [0.75, 0.75, 0.75],
        ]
    )

    result = placement_entropy(points, bins=2)

    assert result["occupied_voxels"] == 4
    assert result["entropy_bits"] == pytest.approx(2.0)
    assert result["normalized_entropy_observed_support"] == pytest.approx(1.0)


def test_mesh_proxies_and_part_hits_have_bounded_interpretable_values() -> None:
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]])
    constellation = np.asarray([[0.0, 0.0, 0.0], [0.9, 0.9, 0.0]])

    proxies = mesh_proxy_values(constellation, vertices, faces)
    hits = part_hit_fractions(
        constellation,
        np.asarray([[-0.05, 0.0, 0.0], [0.9, 0.9, 0.0]]),
        np.asarray(["center", "corner"]),
        radius=0.1,
    )

    assert all(0.0 <= value <= 1.0 for value in proxies["curvature_percentile"])
    assert proxies["extremity_score"][0] == 0.0
    assert hits["fraction_within_radius_by_part"] == {
        "center": 0.5,
        "corner": 0.5,
    }
    assert hits["unassigned_fraction"] == 0.0


def _figure_fixture() -> tuple[
    list[dict[str, object]], dict[tuple[str, str, str, int], np.ndarray], list[str]
]:
    categories = [f"category_{index}" for index in range(8)]
    rows: list[dict[str, object]] = []
    sources = {}
    template = np.asarray(
        [
            [-0.06, -0.03, 0.0],
            [0.06, -0.03, 0.0],
            [-0.06, 0.03, 0.0],
            [0.06, 0.03, 0.0],
        ]
    )
    for category_index, category in enumerate(categories):
        split = "ood" if category_index >= 6 else "validation"
        center = -0.7 + 0.2 * category_index
        for instance_index, jitter in enumerate((-0.005, 0.005)):
            key = (split, category, f"{category}_{instance_index}", instance_index)
            source = template + [center + jitter, 0.0, 0.0]
            sources[key] = np.repeat(source, 8, axis=0)
            for method_index, method in enumerate(METHODS):
                placement = source + [0.0, 0.002 * method_index, 0.0]
                rows.append(
                    {
                        "split": split,
                        "family": category,
                        "model_id": key[2],
                        "sample_id": instance_index,
                        "method": method,
                        "coordinates": placement.tolist(),
                        "nearest_source_sample_distance": [0.002 * method_index] * 4,
                        "curvature_percentile": [0.25] * 4,
                        "symmetry_plane_distance_normalized": [0.1] * 4,
                        "extremity_score": [0.5] * 4,
                    }
                )
    return rows, sources, categories


def test_figure_generators_run_headless_and_summary_covers_all_methods(
    tmp_path: Path,
) -> None:
    rows, sources, categories = _figure_fixture()
    qualitative = tmp_path / "qualitative.svg"
    maps = tmp_path / "maps.svg"

    generate_qualitative_svg(
        rows, sources, qualitative, categories=categories, projection_axes=(0, 2)
    )
    generate_category_maps_svg(rows, maps, bins=4, projection_axes=(0, 2))
    summary = summarize_placement_rows(rows, bins=4)

    assert qualitative.read_text().startswith('<?xml version="1.0"')
    assert "category_7" in qualitative.read_text()
    assert "random best of 16" in qualitative.read_text()
    assert maps.read_text().startswith('<?xml version="1.0"')
    assert summary["category_count"] == 8
    assert summary["cloud_count"] == 16
    assert set(summary["methods"]) == set(METHODS)


def test_config_keeps_adam_64_and_eight_category_figure_fixed() -> None:
    with pytest.raises(ValueError, match="Adam-64"):
        PlacementAnalysisConfig(adam_evaluations=63)
    with pytest.raises(ValueError, match="eight"):
        PlacementAnalysisConfig(figure_categories=("airplane",))


def test_resumed_rows_require_exact_canonical_stream_hashes(tmp_path: Path) -> None:
    stream = encode_constellation(
        np.asarray([[-0.5, 0.0, 0.5], [0.5, 0.0, -0.5]]),
        bits=8,
        mode="fps",
        output_points=8,
    )
    packet = decode_constellation(stream)
    row = {
        "split": "validation",
        "family": "fixture",
        "model_id": "fixture_0",
        "sample_id": 0,
        "method": "fps",
        "stream_hex": stream.hex(),
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "stream_bytes": len(stream),
        "actual_stream_bits": 8 * len(stream),
        "bitstream_mode": packet.mode,
        "coordinate_bits": packet.bits,
        "constellation_size": len(packet.coordinates),
        "coordinates": packet.coordinates.tolist(),
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n")

    assert _load_rows(path) == [row]
    row["stream_sha256"] = "0" * 64
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="exact stream"):
        _load_rows(path)
