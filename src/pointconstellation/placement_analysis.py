"""Experiment 034: semantic placement analysis of coordinate constellations.

The numerical analysis and SVG writers in this module depend only on NumPy and
the standard library.  Torch-backed model inference is imported lazily by the
experiment runner so that placement summaries and figures remain usable in the
base installation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pointconstellation.bitstream import decode_constellation, encode_constellation

METHODS = ("fps", "random_best_of_16", "refiner", "adam_64")
REPRESENTATION_CLASSES = {
    "fps": "strict-subset",
    "random_best_of_16": "strict-subset",
    "refiner": "free-coordinate",
    "adam_64": "free-coordinate",
}
DEFAULT_FIGURE_CATEGORIES = (
    "airplane",
    "car",
    "chair",
    "guitar",
    "lamp",
    "sofa",
    "bed",
    "table",
)
DEFAULT_PART_LABEL_ROOTS = (
    "data/ShapeNetPart",
    "data/shapenet_part_seg_hdf5_data",
    "data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlacementAnalysisConfig:
    """Validated configuration for the fixed-cell placement diagnostic."""

    stability_config: str = "configs/experiment_019_stability_modelnet40.json"
    stability_artifact_dir: str = "artifacts/local/experiment_019_stability_modelnet40"
    decoder_seed: int = 7
    refiner_seed: int = 101
    splits: tuple[str, ...] = ("validation", "ood")
    adam_evaluations: int = 64
    adam_learning_rate: float = 0.03
    selection_seed: int = 20_260_821
    voxel_bins: int = 8
    max_clouds_per_category: int | None = None
    batch_size: int = 4
    figure_categories: tuple[str, ...] = DEFAULT_FIGURE_CATEGORIES
    projection_axes: tuple[int, int] = (0, 2)
    part_label_roots: tuple[str, ...] = DEFAULT_PART_LABEL_ROOTS
    output_dir: str = "artifacts/local/experiment_034_placement_analysis"

    def __post_init__(self) -> None:
        if self.decoder_seed < 0 or self.refiner_seed < 0:
            raise ValueError("model seeds must be nonnegative")
        if (
            not self.splits
            or len(set(self.splits)) != len(self.splits)
            or set(self.splits) - {"validation", "ood"}
        ):
            raise ValueError("splits must be unique validation and/or ood entries")
        if self.adam_evaluations != 64:
            raise ValueError("Experiment 034 requires the predeclared Adam-64 budget")
        if self.adam_learning_rate <= 0:
            raise ValueError("adam_learning_rate must be positive")
        if not 0 <= self.selection_seed < 2**63:
            raise ValueError("selection_seed must be a nonnegative 63-bit integer")
        if not 2 <= self.voxel_bins <= 64:
            raise ValueError("voxel_bins must be between 2 and 64")
        if (
            self.max_clouds_per_category is not None
            and self.max_clouds_per_category < 2
        ):
            raise ValueError("max_clouds_per_category must be at least two")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(set(self.figure_categories)) != len(self.figure_categories):
            raise ValueError("figure_categories must be unique")
        if len(self.figure_categories) not in {0, 8}:
            raise ValueError("figure_categories must be empty or contain eight names")
        if (
            len(self.projection_axes) != 2
            or len(set(self.projection_axes)) != 2
            or set(self.projection_axes) - {0, 1, 2}
        ):
            raise ValueError("projection_axes must contain two distinct xyz axes")

    @classmethod
    def from_json(cls, path: Path) -> PlacementAnalysisConfig:
        values = json.loads(path.read_text())
        for key in (
            "splits",
            "figure_categories",
            "projection_axes",
            "part_label_roots",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def _points(value: ArrayLike, *, name: str, minimum: int = 1) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"{name} must have shape (N, 3) with N >= {minimum}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def symmetric_nearest_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Return symmetric mean nearest-neighbour Euclidean set distance."""

    first_points = _points(first, name="first")
    second_points = _points(second, name="second")
    distances = np.linalg.norm(
        first_points[:, None, :] - second_points[None, :, :], axis=2
    )
    return 0.5 * float(distances.min(axis=1).mean() + distances.min(axis=0).mean())


def category_consistency(
    constellations: ArrayLike,
    categories: Sequence[str],
    instance_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare unordered constellations within and across semantic categories."""

    values = np.asarray(constellations, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3 or not len(values):
        raise ValueError("constellations must have shape (C, K, 3)")
    if not np.isfinite(values).all() or len(categories) != len(values):
        raise ValueError("categories must align with finite constellations")
    identities = (
        tuple(str(index) for index in range(len(values)))
        if instance_ids is None
        else tuple(str(value) for value in instance_ids)
    )
    if len(identities) != len(values):
        raise ValueError("instance_ids must align with constellations")
    category_names = tuple(str(value) for value in categories)
    if len(set(category_names)) < 2:
        raise ValueError("category consistency requires at least two categories")

    within: list[float] = []
    across: list[float] = []
    per_category: dict[str, dict[str, list[float]]] = {
        category: {"within": [], "across": []}
        for category in sorted(set(category_names))
    }
    for first in range(len(values)):
        for second in range(first + 1, len(values)):
            if identities[first] == identities[second]:
                continue
            distance = symmetric_nearest_distance(values[first], values[second])
            first_category = category_names[first]
            second_category = category_names[second]
            if first_category == second_category:
                within.append(distance)
                per_category[first_category]["within"].append(distance)
            else:
                across.append(distance)
                per_category[first_category]["across"].append(distance)
                per_category[second_category]["across"].append(distance)
    if not within or not across:
        raise ValueError("consistency requires within- and across-category pairs")

    within_mean = float(np.mean(within))
    across_mean = float(np.mean(across))
    ratio = within_mean / max(across_mean, 1e-12)
    return {
        "distance_definition": (
            "symmetric mean nearest-neighbour Euclidean distance between unordered "
            "constellation sets in the declared canonical frame"
        ),
        "within_category_mean": within_mean,
        "across_category_mean": across_mean,
        "within_to_across_ratio": ratio,
        "relative_separation": (across_mean - within_mean) / max(across_mean, 1e-12),
        "within_pair_count": len(within),
        "across_pair_count": len(across),
        "per_category": {
            category: {
                "within_category_mean": float(np.mean(groups["within"])),
                "across_category_mean": float(np.mean(groups["across"])),
                "within_pair_count": len(groups["within"]),
                "across_pair_count": len(groups["across"]),
            }
            for category, groups in per_category.items()
            if groups["within"] and groups["across"]
        },
    }


def placement_entropy(points: ArrayLike, *, bins: int) -> dict[str, Any]:
    """Measure discrete occupancy entropy on a fixed ``[-1, 1]^3`` grid."""

    values = _points(points, name="points")
    if not 2 <= bins <= 64:
        raise ValueError("bins must be between 2 and 64")
    tolerance = 1e-9
    if np.any(values < -1.0 - tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("placement points must lie in [-1, 1]")
    clipped = np.clip(values, -1.0, 1.0)
    histogram, _ = np.histogramdd(
        clipped,
        bins=(bins, bins, bins),
        range=((-1.0, 1.0),) * 3,
    )
    occupied = np.argwhere(histogram > 0)
    counts = histogram[tuple(occupied.T)].astype(np.int64)
    probabilities = counts / counts.sum()
    entropy_bits = float(-(probabilities * np.log2(probabilities)).sum())
    volume_maximum = math.log2(bins**3)
    observed_maximum = math.log2(min(len(values), bins**3))
    return {
        "grid_bins_per_axis": bins,
        "point_count": len(values),
        "occupied_voxels": len(occupied),
        "entropy_bits": entropy_bits,
        "normalized_entropy_grid_volume": entropy_bits / volume_maximum,
        "normalized_entropy_observed_support": (
            entropy_bits / observed_maximum if observed_maximum > 0 else 0.0
        ),
        "sparse_voxel_counts": [
            {"voxel": index.tolist(), "count": int(count)}
            for index, count in zip(occupied, counts, strict=True)
        ],
    }


def nearest_point_distances(points: ArrayLike, reference: ArrayLike) -> FloatArray:
    """Return distances to a finite reference sample, not to a surface."""

    queries = _points(points, name="points")
    targets = _points(reference, name="reference")
    result = np.full(len(queries), np.inf, dtype=np.float64)
    for start in range(0, len(targets), 2048):
        chunk = targets[start : start + 2048]
        squared = ((queries[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        result = np.minimum(result, squared.min(axis=1))
    return np.sqrt(result)


def vertex_curvature_proxy(vertices: ArrayLike, faces: ArrayLike) -> FloatArray:
    """Return area-weighted incident-face normal dispersion at each vertex."""

    vertex_values = _points(vertices, name="vertices", minimum=3)
    face_values = np.asarray(faces, dtype=np.int64)
    if face_values.ndim != 2 or face_values.shape[1] != 3 or not len(face_values):
        raise ValueError("faces must have shape (F, 3)")
    if face_values.min() < 0 or face_values.max() >= len(vertex_values):
        raise ValueError("faces contain an out-of-range vertex index")
    triangles = vertex_values[face_values]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]
    accumulated = np.zeros_like(vertex_values)
    total_area = np.zeros(len(vertex_values), dtype=np.float64)
    weighted = normals * double_area[:, None]
    for corner in range(3):
        np.add.at(accumulated, face_values[:, corner], weighted)
        np.add.at(total_area, face_values[:, corner], double_area)
    coherence = np.linalg.norm(accumulated, axis=1) / np.maximum(total_area, 1e-12)
    return np.clip(1.0 - coherence, 0.0, 1.0)


def _midrank_percentiles(values: FloatArray, selected: FloatArray) -> FloatArray:
    lower = (values[:, None] < selected[None, :]).sum(axis=0)
    equal = np.isclose(values[:, None], selected[None, :], atol=1e-12).sum(axis=0)
    return (lower + 0.5 * equal) / len(values)


def _symmetry_axis(vertices: FloatArray) -> tuple[int, tuple[float, float, float]]:
    if len(vertices) > 1024:
        indices = np.linspace(0, len(vertices) - 1, 1024, dtype=np.int64)
        sample = vertices[indices]
    else:
        sample = vertices
    scores = []
    for axis in range(3):
        reflected = sample.copy()
        reflected[:, axis] *= -1.0
        scores.append(float(nearest_point_distances(reflected, sample).mean()))
    return int(np.argmin(scores)), tuple(scores)


@dataclass(frozen=True)
class MeshProxyContext:
    """Per-mesh quantities shared by every placement method."""

    vertices: FloatArray
    curvature: FloatArray
    symmetry_axis: int
    symmetry_scores: tuple[float, float, float]
    plane_scale: float
    radial_scale: float


def mesh_proxy_context(vertices: ArrayLike, faces: ArrayLike) -> MeshProxyContext:
    """Precompute deterministic mesh quantities used by placement proxies."""

    vertex_values = _points(vertices, name="vertices", minimum=3)
    curvature = vertex_curvature_proxy(vertex_values, faces)
    axis, symmetry_scores = _symmetry_axis(vertex_values)
    return MeshProxyContext(
        vertices=vertex_values,
        curvature=curvature,
        symmetry_axis=axis,
        symmetry_scores=symmetry_scores,
        plane_scale=max(float(np.abs(vertex_values[:, axis]).max()), 1e-12),
        radial_scale=max(float(np.linalg.norm(vertex_values, axis=1).max()), 1e-12),
    )


def mesh_proxy_values(
    constellation: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike | None,
    *,
    context: MeshProxyContext | None = None,
) -> dict[str, Any]:
    """Evaluate curvature, canonical-plane, and radial-extremity proxies."""

    coordinates = _points(constellation, name="constellation")
    if context is None:
        if faces is None:
            raise ValueError("faces are required when context is not provided")
        context = mesh_proxy_context(vertices, faces)
    vertex_values = context.vertices
    squared = ((coordinates[:, None, :] - vertex_values[None, :, :]) ** 2).sum(axis=2)
    nearest_vertices = squared.argmin(axis=1)
    curvature_percentile = _midrank_percentiles(
        context.curvature, context.curvature[nearest_vertices]
    )
    return {
        "curvature_percentile": curvature_percentile.tolist(),
        "symmetry_axis": context.symmetry_axis,
        "symmetry_axis_reflection_sample_distance": list(context.symmetry_scores),
        "symmetry_plane_distance_normalized": (
            np.abs(coordinates[:, context.symmetry_axis]) / context.plane_scale
        ).tolist(),
        "extremity_score": (
            np.linalg.norm(coordinates, axis=1) / context.radial_scale
        ).tolist(),
    }


def part_hit_fractions(
    constellation: ArrayLike,
    labelled_points: ArrayLike,
    part_labels: ArrayLike,
    *,
    radius: float,
) -> dict[str, Any]:
    """Return the fraction of constellation points within ``radius`` of each part."""

    coordinates = _points(constellation, name="constellation")
    labelled = _points(labelled_points, name="labelled_points")
    labels = np.asarray(part_labels)
    if labels.ndim != 1 or len(labels) != len(labelled):
        raise ValueError("part_labels must align with labelled_points")
    if radius <= 0:
        raise ValueError("radius must be positive")
    hit_any = np.zeros(len(coordinates), dtype=bool)
    fractions: dict[str, float] = {}
    for label in sorted(set(labels.tolist()), key=str):
        distances = nearest_point_distances(coordinates, labelled[labels == label])
        hits = distances <= radius
        hit_any |= hits
        fractions[str(label)] = float(hits.mean())
    return {
        "radius": radius,
        "fraction_within_radius_by_part": fractions,
        "unassigned_fraction": float((~hit_any).mean()),
    }


def _record_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["split"]),
        str(row["family"]),
        str(row["model_id"]),
        int(row["sample_id"]),
    )


def _method_rows(
    rows: Sequence[Mapping[str, Any]], method: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if row["method"] == method]
    selected.sort(key=_record_key)
    return selected


def _mean_proxy(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    fields = (
        "nearest_source_sample_distance",
        "curvature_percentile",
        "symmetry_plane_distance_normalized",
        "extremity_score",
    )
    result = {}
    for field in fields:
        values = np.concatenate(
            [np.asarray(row[field], dtype=np.float64) for row in rows]
        )
        result[f"mean_{field}"] = float(values.mean())
    curvature = np.concatenate(
        [np.asarray(row["curvature_percentile"], dtype=np.float64) for row in rows]
    )
    symmetry = np.concatenate(
        [
            np.asarray(row["symmetry_plane_distance_normalized"], dtype=np.float64)
            for row in rows
        ]
    )
    extremity = np.concatenate(
        [np.asarray(row["extremity_score"], dtype=np.float64) for row in rows]
    )
    result.update(
        {
            "fraction_curvature_percentile_at_least_0_9": float(
                (curvature >= 0.9).mean()
            ),
            "fraction_within_0_05_of_symmetry_plane": float((symmetry <= 0.05).mean()),
            "fraction_extremity_at_least_0_9": float((extremity >= 0.9).mean()),
        }
    )
    return result


def summarize_placement_rows(
    rows: Sequence[Mapping[str, Any]], *, bins: int
) -> dict[str, Any]:
    """Create category maps, consistency metrics, and proxy summaries."""

    if not rows:
        raise ValueError("placement rows cannot be empty")
    keys = [(_record_key(row), str(row["method"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("placement rows contain duplicate cloud/method keys")
    clouds = sorted({_record_key(row) for row in rows})
    expected = {(cloud, method) for cloud in clouds for method in METHODS}
    if set(keys) != expected:
        raise ValueError("placement rows do not form the complete four-method grid")
    categories = sorted({str(row["family"]) for row in rows})
    method_summaries: dict[str, Any] = {}
    per_category: dict[str, Any] = {}
    for method in METHODS:
        selected = _method_rows(rows, method)
        constellations = np.asarray([row["coordinates"] for row in selected])
        method_summaries[method] = {
            "representation_class": REPRESENTATION_CLASSES[method],
            "category_consistency": category_consistency(
                constellations,
                [str(row["family"]) for row in selected],
                [str(row["model_id"]) for row in selected],
            ),
            "mesh_proxy_summary": _mean_proxy(selected),
        }
    for category in categories:
        category_rows = [row for row in rows if row["family"] == category]
        per_category[category] = {
            "split": str(category_rows[0]["split"]),
            "is_category_ood": category_rows[0]["split"] == "ood",
            "instance_count": len({_record_key(row) for row in category_rows}),
            "methods": {},
        }
        for method in METHODS:
            selected = [row for row in category_rows if row["method"] == method]
            points = np.concatenate(
                [np.asarray(row["coordinates"], dtype=np.float64) for row in selected]
            )
            per_category[category]["methods"][method] = {
                "placement_map": placement_entropy(points, bins=bins),
                "mesh_proxy_summary": _mean_proxy(selected),
            }
    control_ratio = min(
        method_summaries[method]["category_consistency"]["within_to_across_ratio"]
        for method in ("fps", "random_best_of_16")
    )
    learned_ratios = {
        method: method_summaries[method]["category_consistency"][
            "within_to_across_ratio"
        ]
        for method in ("refiner", "adam_64")
    }
    return {
        "category_count": len(categories),
        "cloud_count": len(clouds),
        "method_order": list(METHODS),
        "methods": method_summaries,
        "per_category": per_category,
        "h_b4_gate": {
            "criterion": (
                "both free-coordinate methods have a lower within/across set-"
                "distance ratio than both strict-subset controls"
            ),
            "strict_subset_best_ratio": control_ratio,
            "free_coordinate_ratios": learned_ratios,
            "passes": all(value < control_ratio for value in learned_ratios.values()),
        },
    }


def _svg_color(value: float, maximum: float) -> str:
    scaled = min(max(value / max(maximum, 1e-12), 0.0), 1.0)
    red = round(35 + 220 * scaled)
    green = round(105 + 75 * (1.0 - scaled))
    blue = round(220 - 180 * scaled)
    return f"rgb({red},{green},{blue})"


def _project(
    points: FloatArray,
    axes: tuple[int, int],
    *,
    left: float,
    top: float,
    size: float,
) -> tuple[FloatArray, FloatArray]:
    x = left + (np.clip(points[:, axes[0]], -1.0, 1.0) + 1.0) * 0.5 * size
    y = top + (1.0 - (np.clip(points[:, axes[1]], -1.0, 1.0) + 1.0) * 0.5) * size
    return x, y


def generate_qualitative_svg(
    rows: Sequence[Mapping[str, Any]],
    sources: Mapping[tuple[str, str, str, int], ArrayLike],
    output_path: Path,
    *,
    categories: Sequence[str],
    projection_axes: tuple[int, int] = (0, 2),
) -> None:
    """Write a headless source/constellation overlay with sample-distance colors."""

    if len(categories) != 8 or len(set(categories)) != 8:
        raise ValueError("qualitative figure requires eight unique categories")
    selected_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    selected_sources: dict[str, FloatArray] = {}
    split_by_category: dict[str, str] = {}
    for category in categories:
        candidates = sorted(
            {_record_key(row) for row in rows if row["family"] == category}
        )
        if not candidates:
            raise ValueError(f"figure category is absent: {category}")
        cloud = candidates[0]
        if cloud not in sources:
            raise ValueError(f"source points are absent for figure cloud: {cloud}")
        selected_sources[category] = _points(sources[cloud], name="source")
        split_by_category[category] = cloud[0]
        for method in METHODS:
            matches = [
                row
                for row in rows
                if _record_key(row) == cloud and row["method"] == method
            ]
            if len(matches) != 1:
                raise ValueError("figure requires one row per cloud and method")
            selected_rows[(category, method)] = matches[0]
    if not any(split_by_category[category] == "ood" for category in categories):
        raise ValueError("qualitative figure must include at least one OOD category")

    all_errors = np.concatenate(
        [
            np.asarray(row["nearest_source_sample_distance"], dtype=np.float64)
            for row in selected_rows.values()
        ]
    )
    color_maximum = max(float(np.quantile(all_errors, 0.95)), 1e-6)
    panel = 122
    gap = 18
    label_width = 104
    width = label_width + 5 * (panel + gap) + 40
    height = 70 + len(categories) * (panel + 28) + 45
    labels = ("source", *METHODS)
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}</style>",
        '<text x="18" y="25" font-size="16" font-weight="bold">'
        "Experiment 034: canonical-frame placement and source-sample error</text>",
    ]
    for column, label in enumerate(labels):
        x = label_width + column * (panel + gap) + panel / 2
        elements.append(
            f'<text x="{x:.1f}" y="52" text-anchor="middle" font-size="12">'
            f"{html.escape(label.replace('_', ' '))}</text>"
        )
    for row_index, category in enumerate(categories):
        top = 65 + row_index * (panel + 28)
        badge = "OOD" if split_by_category[category] == "ood" else "ID"
        elements.append(
            f'<text x="10" y="{top + panel / 2:.1f}" font-size="12">'
            f"{html.escape(category)}</text>"
        )
        elements.append(
            f'<text x="10" y="{top + panel / 2 + 16:.1f}" font-size="9" '
            f'fill="#666">{badge}</text>'
        )
        source = selected_sources[category]
        display_source = source[:: max(1, math.ceil(len(source) / 350))]
        for column, label in enumerate(labels):
            left = label_width + column * (panel + gap)
            elements.append(
                f'<rect x="{left}" y="{top}" width="{panel}" height="{panel}" '
                'fill="#fafafa" stroke="#c7c7c7"/>'
            )
            x, y = _project(
                display_source,
                projection_axes,
                left=left,
                top=top,
                size=panel,
            )
            elements.extend(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="0.75" fill="#b9b9b9"/>'
                for px, py in zip(x, y, strict=True)
            )
            if label == "source":
                continue
            placement = _points(
                selected_rows[(category, label)]["coordinates"],
                name="coordinates",
            )
            errors = np.asarray(
                selected_rows[(category, label)]["nearest_source_sample_distance"],
                dtype=np.float64,
            )
            x, y = _project(
                placement,
                projection_axes,
                left=left,
                top=top,
                size=panel,
            )
            elements.extend(
                (
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.0" '
                    f'fill="{_svg_color(error, color_maximum)}" '
                    'stroke="white" stroke-width="0.8"/>'
                )
                for px, py, error in zip(x, y, errors, strict=True)
            )
    legend_y = height - 24
    elements.append(
        f'<text x="{label_width}" y="{legend_y}" font-size="10">nearest source-sample '
        "distance (blue=0, red="
        f"{color_maximum:.3g}); this is not continuous-surface error</text>"
    )
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n")


def generate_category_maps_svg(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    bins: int,
    projection_axes: tuple[int, int] = (0, 2),
) -> None:
    """Write projected aggregate placement maps for every available category."""

    categories = sorted({str(row["family"]) for row in rows})
    cell = 62
    gap = 14
    label_width = 112
    width = label_width + len(METHODS) * (cell + gap) + 24
    height = 54 + len(categories) * (cell + 12) + 18
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}</style>",
        '<text x="10" y="20" font-size="14" font-weight="bold">'
        "Experiment 034 category placement maps</text>",
    ]
    for column, method in enumerate(METHODS):
        x = label_width + column * (cell + gap) + cell / 2
        elements.append(
            f'<text x="{x:.1f}" y="42" text-anchor="middle" font-size="9">'
            f"{html.escape(method.replace('_', ' '))}</text>"
        )
    for row_index, category in enumerate(categories):
        top = 49 + row_index * (cell + 12)
        category_rows = [row for row in rows if row["family"] == category]
        is_ood = category_rows[0]["split"] == "ood"
        suffix = " (OOD)" if is_ood else ""
        elements.append(
            f'<text x="8" y="{top + cell / 2:.1f}" font-size="9">'
            f"{html.escape(category + suffix)}</text>"
        )
        for column, method in enumerate(METHODS):
            left = label_width + column * (cell + gap)
            points = np.concatenate(
                [
                    np.asarray(row["coordinates"], dtype=np.float64)
                    for row in category_rows
                    if row["method"] == method
                ]
            )
            histogram, _, _ = np.histogram2d(
                points[:, projection_axes[0]],
                points[:, projection_axes[1]],
                bins=bins,
                range=((-1.0, 1.0), (-1.0, 1.0)),
            )
            maximum = max(float(histogram.max()), 1.0)
            pixel = cell / bins
            elements.append(
                f'<rect x="{left}" y="{top}" width="{cell}" height="{cell}" '
                'fill="#fafafa" stroke="#c7c7c7"/>'
            )
            for x_index, y_index in np.argwhere(histogram > 0):
                intensity = histogram[x_index, y_index] / maximum
                shade = round(245 - 190 * math.sqrt(float(intensity)))
                elements.append(
                    f'<rect x="{left + x_index * pixel:.2f}" '
                    f'y="{top + (bins - 1 - y_index) * pixel:.2f}" '
                    f'width="{pixel + 0.1:.2f}" height="{pixel + 0.1:.2f}" '
                    f'fill="rgb({shade},{shade},{255})"/>'
                )
    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n")


def _selection_seed(
    config: PlacementAnalysisConfig,
    *,
    split: str,
    family: str,
    model_id: str,
    sample_id: int,
) -> int:
    identity = json.dumps(
        [
            config.selection_seed,
            "random_best_of_n",
            split,
            family,
            model_id,
            sample_id,
        ],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (
        2**63
    )


def _selected_dataset_indices(
    datasets: Mapping[str, Any], config: PlacementAnalysisConfig
) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    for split in config.splits:
        counts: dict[str, int] = {}
        for index in range(len(datasets[split])):
            record = datasets[split].records[index]
            category = str(record["category"])
            if (
                config.max_clouds_per_category is not None
                and counts.get(category, 0) >= config.max_clouds_per_category
            ):
                continue
            selected.append((split, index))
            counts[category] = counts.get(category, 0) + 1
    return selected


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(_record_key(row), str(row["method"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("placement artifact contains duplicate rows")
    for row in rows:
        method = str(row["method"])
        if method not in METHODS:
            raise RuntimeError(f"placement artifact has an unknown method: {method}")
        stream = bytes.fromhex(str(row["stream_hex"]))
        packet = decode_constellation(stream)
        coordinates = np.asarray(row["coordinates"], dtype=np.float64)
        if (
            hashlib.sha256(stream).hexdigest() != row["stream_sha256"]
            or len(stream) != row["stream_bytes"]
            or 8 * len(stream) != row["actual_stream_bits"]
            or packet.mode != row["bitstream_mode"]
            or packet.bits != row["coordinate_bits"]
            or len(packet.coordinates) != row["constellation_size"]
            or not np.array_equal(packet.coordinates, coordinates)
            or encode_constellation(
                coordinates,
                bits=packet.bits,
                mode=packet.mode,
                output_points=packet.output_points,
            )
            != stream
        ):
            raise RuntimeError("placement artifact failed exact stream validation")
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _part_label_status(config: PlacementAnalysisConfig) -> dict[str, Any]:
    paths = [Path(value) for value in config.part_label_roots]
    present = [str(path) for path in paths if path.exists()]
    return {
        "status": "not_used_in_modelnet40_analysis",
        "searched_local_roots": [str(path) for path in paths],
        "existing_roots": present,
        "reason": (
            "no compatible part-labelled subset was configured"
            if not present
            else "part labels do not share the evaluated ModelNet40 instance IDs"
        ),
        "fallback": "mesh_geometry_proxies",
    }


def _figure_categories(
    rows: Sequence[Mapping[str, Any]], configured: Sequence[str]
) -> tuple[str, ...]:
    available = sorted({str(row["family"]) for row in rows})
    if configured:
        missing = set(configured) - set(available)
        if missing:
            raise ValueError(
                f"configured figure categories are absent: {sorted(missing)}"
            )
        result = tuple(configured)
    else:
        id_categories = [
            category
            for category in available
            if any(row["family"] == category and row["split"] != "ood" for row in rows)
        ]
        ood_categories = [
            category
            for category in available
            if any(row["family"] == category and row["split"] == "ood" for row in rows)
        ]
        result = tuple(id_categories[:6] + ood_categories[:2])
    if len(result) != 8:
        raise ValueError("eight categories, including OOD, are required for the figure")
    return result


def run_placement_analysis(
    config: PlacementAnalysisConfig, *, device_name: str | None = None
) -> dict[str, Any]:
    """Regenerate exact placements from one sealed Experiment 019 model cell."""

    # Imports stay local so pure NumPy summaries/figures do not require Torch.
    import torch

    from pointconstellation.data import file_sha256
    from pointconstellation.headroom_experiment import (
        _batch_tensors,
        _search_adam_start,
        _source_scorer,
    )
    from pointconstellation.official_stability import (
        OfficialStabilityConfig,
        _load_models,
    )
    from pointconstellation.refiner_experiment import _state_hash
    from pointconstellation.selection_baselines import SELECTION_METHODS
    from pointconstellation.stability_experiment import (
        StabilityExperimentConfig,
        _data_protocol,
        _datasets,
        _per_cloud_chamfer,
    )
    from pointconstellation.train import select_device

    stability_path = Path(config.stability_config)
    stability = StabilityExperimentConfig.from_json(stability_path)
    if stability.dataset_kind != "mesh_manifest":
        raise ValueError("Experiment 034 requires the canonical mesh-manifest dataset")
    if config.decoder_seed not in stability.decoder_seeds:
        raise ValueError("decoder_seed is absent from Experiment 019")
    if config.refiner_seed not in stability.refiner_seeds:
        raise ValueError("refiner_seed is absent from Experiment 019")
    artifact_dir = Path(config.stability_artifact_dir)
    stability_metrics_path = artifact_dir / "stability_metrics.json"
    stability_metrics = json.loads(stability_metrics_path.read_text())
    if stability_metrics["config"] != json.loads(json.dumps(asdict(stability))):
        raise RuntimeError("Experiment 019 artifact config differs from checked config")
    if not all(stability_metrics["contract_checks"].values()):
        raise RuntimeError("Experiment 019 artifact has a failed contract check")

    datasets = _datasets(stability)
    data_protocol = _data_protocol(stability, datasets)
    if data_protocol != stability_metrics["data_protocol"]:
        raise RuntimeError("Experiment 019 data identity changed before placement run")
    device = select_device(device_name)
    official = OfficialStabilityConfig(
        stability_config=config.stability_config,
        stability_artifact_dir=config.stability_artifact_dir,
        position_bits=stability.coordinate_bits,
        decoder_seeds=stability.decoder_seeds,
        refiner_seeds=stability.refiner_seeds,
        splits=config.splits,
    )
    decoder, refiner, model_metadata = _load_models(
        stability,
        official,
        decoder_seed=config.decoder_seed,
        refiner_seed=config.refiner_seed,
        device=device,
    )
    assert refiner is not None
    decoder_hash_before = _state_hash(decoder)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "placement_points.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "experiment": "034_placement_analysis",
        "config": json.loads(json.dumps(asdict(config))),
        "stability_config_sha256": file_sha256(stability_path),
        "stability_metrics_sha256": file_sha256(stability_metrics_path),
        "data_protocol": data_protocol,
        "model_artifacts": model_metadata,
        "canonical_frame": (
            "mesh bounding-box center followed by unit maximum-vertex-radius scale; "
            "no post-hoc instance rotation or reflection"
        ),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing Experiment 034 manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows = _load_rows(rows_path)
    completed = {(_record_key(row), str(row["method"])) for row in rows}
    selected_indices = _selected_dataset_indices(datasets, config)
    for batch_start in range(0, len(selected_indices), config.batch_size):
        batch_indices = selected_indices[batch_start : batch_start + config.batch_size]
        samples = [datasets[split][index] for split, index in batch_indices]
        metadata = [
            {
                "split": split,
                "family": str(sample["family"]),
                "model_id": str(sample["model_id"]),
                "sample_id": int(sample["sample_id"]),
                "dataset_index": index,
            }
            for (split, index), sample in zip(batch_indices, samples, strict=True)
        ]
        expected_batch = {
            (
                (
                    item["split"],
                    item["family"],
                    item["model_id"],
                    item["sample_id"],
                ),
                method,
            )
            for item in metadata
            for method in METHODS
        }
        if expected_batch <= completed:
            continue
        source, _, _ = _batch_tensors(samples, device)
        fps = torch.stack(
            [
                SELECTION_METHODS["fps"](
                    cloud,
                    stability.constellation_size,
                    stability.coordinate_bits,
                    0,
                    None,
                )
                for cloud in source
            ]
        )
        random_best = []
        for cloud, item in zip(source, metadata, strict=True):

            def scorer(candidate: Any, target: Any = cloud) -> float:
                with torch.no_grad():
                    reconstruction = decoder(
                        candidate.unsqueeze(0),
                        num_output_points=stability.num_points,
                    )
                    loss = _per_cloud_chamfer(
                        reconstruction,
                        target.unsqueeze(0),
                        chunk_size=stability.distance_chunk_size,
                    )
                return float(loss.item())

            seed = _selection_seed(
                config,
                **{
                    key: item[key]
                    for key in ("split", "family", "model_id", "sample_id")
                },
            )
            random_best.append(
                SELECTION_METHODS["random_best_of_16"](
                    cloud,
                    stability.constellation_size,
                    stability.coordinate_bits,
                    seed,
                    scorer,
                )
            )
        random_best_tensor = torch.stack(random_best)
        refiner_coordinates = refiner(
            source,
            stability.constellation_size,
            decoder=decoder,
            target=source,
            num_output_points=stability.num_points,
        )
        source_score = _source_scorer(
            decoder,
            source,
            num_output_points=stability.num_points,
            chunk_size=stability.distance_chunk_size,
        )
        adam_result = _search_adam_start(
            source_score,
            fps,
            bits=stability.coordinate_bits,
            budget=config.adam_evaluations,
            learning_rate=config.adam_learning_rate,
        )
        method_coordinates = {
            "fps": fps,
            "random_best_of_16": random_best_tensor,
            "refiner": refiner_coordinates,
            "adam_64": adam_result.coordinates,
        }
        proxy_contexts = []
        for item in metadata:
            dataset = datasets[item["split"]]
            mesh = dataset._load_mesh(dataset.records[item["dataset_index"]])
            proxy_contexts.append(mesh_proxy_context(mesh.vertices, mesh.faces))
        for method, coordinates in method_coordinates.items():
            mode = (
                "fps"
                if method == "fps"
                else "strict_subset"
                if method == "random_best_of_16"
                else "free"
            )
            for cloud_index, (sample, item, proxy_context) in enumerate(
                zip(samples, metadata, proxy_contexts, strict=True)
            ):
                key = (
                    (
                        item["split"],
                        item["family"],
                        item["model_id"],
                        item["sample_id"],
                    ),
                    method,
                )
                if key in completed:
                    continue
                raw = coordinates[cloud_index].detach().cpu().numpy()
                stream = encode_constellation(
                    raw,
                    bits=stability.coordinate_bits,
                    mode=mode,
                    output_points=stability.num_points,
                )
                packet = decode_constellation(stream)
                if (
                    encode_constellation(
                        packet.coordinates,
                        bits=packet.bits,
                        mode=packet.mode,
                        output_points=packet.output_points,
                    )
                    != stream
                ):
                    raise RuntimeError("placement stream failed canonical round trip")
                source_points = sample["source_points"].detach().cpu().numpy()
                proxies = mesh_proxy_values(
                    packet.coordinates,
                    proxy_context.vertices,
                    None,
                    context=proxy_context,
                )
                row = {
                    "experiment": "034_placement_analysis",
                    "split": item["split"],
                    "family": item["family"],
                    "model_id": item["model_id"],
                    "sample_id": item["sample_id"],
                    "method": method,
                    "representation_class": REPRESENTATION_CLASSES[method],
                    "decoder_seed": config.decoder_seed,
                    "refiner_seed": (
                        config.refiner_seed if method == "refiner" else None
                    ),
                    "source_only_optimization": True,
                    "adam_decoder_evaluations": (
                        config.adam_evaluations if method == "adam_64" else None
                    ),
                    "selection_trials": 16 if method == "random_best_of_16" else None,
                    "constellation_size": len(packet.coordinates),
                    "coordinate_bits": packet.bits,
                    "bitstream_mode": packet.mode,
                    "stream_hex": stream.hex(),
                    "stream_sha256": hashlib.sha256(stream).hexdigest(),
                    "stream_bytes": len(stream),
                    "actual_stream_bits": 8 * len(stream),
                    "actual_stream_bpp": 8.0 * len(stream) / stability.num_points,
                    "coordinates": packet.coordinates.tolist(),
                    "nearest_source_sample_distance": nearest_point_distances(
                        packet.coordinates, source_points
                    ).tolist(),
                    **proxies,
                }
                _append_row(rows_path, row)
                rows.append(row)
                completed.add(key)

    if _state_hash(decoder) != decoder_hash_before:
        raise RuntimeError("frozen decoder changed during placement inference")
    expected_rows = {
        (
            (
                split,
                str(datasets[split].records[index]["category"]),
                str(datasets[split].records[index]["model_id"]),
                index,
            ),
            method,
        )
        for split, index in selected_indices
        for method in METHODS
    }
    actual_rows = {(_record_key(row), str(row["method"])) for row in rows}
    if actual_rows != expected_rows:
        raise RuntimeError("placement rows do not match the selected dataset grid")
    analysis = summarize_placement_rows(rows, bins=config.voxel_bins)
    categories = _figure_categories(rows, config.figure_categories)
    sources: dict[tuple[str, str, str, int], Any] = {}
    requested = set(categories)
    for split, index in selected_indices:
        sample = datasets[split][index]
        if str(sample["family"]) not in requested:
            continue
        key = (
            split,
            str(sample["family"]),
            str(sample["model_id"]),
            int(sample["sample_id"]),
        )
        sources[key] = sample["source_points"].detach().cpu().numpy()
    qualitative_path = output_dir / "qualitative_overlays.svg"
    maps_path = output_dir / "category_placement_maps.svg"
    generate_qualitative_svg(
        rows,
        sources,
        qualitative_path,
        categories=categories,
        projection_axes=config.projection_axes,
    )
    generate_category_maps_svg(
        rows,
        maps_path,
        bins=config.voxel_bins,
        projection_axes=config.projection_axes,
    )
    result = {
        "experiment": "034_placement_analysis",
        "status": "complete",
        "config": json.loads(json.dumps(asdict(config))),
        "device": str(device),
        "analysis_scope": (
            "descriptive single sealed decoder/refiner cell; not a multi-seed or "
            "rate-distortion compression claim"
        ),
        "canonical_frame": manifest["canonical_frame"],
        "part_hit_analysis": _part_label_status(config),
        "analysis": analysis,
        "figure_categories": list(categories),
        "outputs": {
            "placement_points": str(rows_path),
            "qualitative_figure": str(qualitative_path),
            "category_placement_maps": str(maps_path),
        },
        "contract_checks": {
            "complete_message_is_quantized_unordered_k_by_3": True,
            "all_stream_round_trips_exact": True,
            "source_only_optimization": True,
            "decoder_frozen": True,
            "canonical_frame_has_no_posthoc_alignment": True,
            "nearest_source_distance_labelled_as_sample_proxy": True,
            "complete_four_method_grid": True,
        },
    }
    summary_path = output_dir / "placement_summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_034_placement_analysis.json"),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    result = run_placement_analysis(
        PlacementAnalysisConfig.from_json(args.config),
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary_path": result["summary_path"],
                "categories": result["analysis"]["category_count"],
                "clouds": result["analysis"]["cloud_count"],
                "h_b4_gate_passes": result["analysis"]["h_b4_gate"]["passes"],
            }
        )
    )


if __name__ == "__main__":
    main()
