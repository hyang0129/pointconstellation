"""Bjøntegaard metrics and hierarchical uncertainty for registry RD curves."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

MINIMUM_BD_POINTS = 4


@dataclass(frozen=True)
class BDResult:
    """One gated Bjøntegaard estimate.

    ``value`` is ``None`` exactly when ``reason`` explains why interpolation was
    not admissible. BD-rate values are percentages; BD-PSNR values are dB.
    """

    value: float | None
    reason: str | None
    overlap_min: float | None = None
    overlap_max: float | None = None
    anchor_points: int = 0
    test_points: int = 0


@dataclass(frozen=True)
class BDBootstrapEstimate:
    """Point estimate and hierarchical-bootstrap confidence interval."""

    value: float | None
    ci_lower: float | None
    ci_upper: float | None
    reason: str | None
    bootstrap_valid: int
    bootstrap_samples: int
    overlap_min: float | None = None
    overlap_max: float | None = None


@dataclass(frozen=True)
class BDComparison:
    """BD-rate and BD-PSNR for a test curve relative to an anchor curve."""

    bd_rate: BDBootstrapEstimate
    bd_psnr: BDBootstrapEstimate
    paired_clouds: int


@dataclass(frozen=True)
class RDCurvePoint:
    """A measured aggregate rate point with seed/cloud uncertainty."""

    rate_point: str
    rate: float
    quality: float
    ci_lower: float | None
    ci_upper: float | None
    dominated: bool
    observations: int


@dataclass(frozen=True)
class RDCurve:
    """Ordered measured points for one method."""

    points: tuple[RDCurvePoint, ...]
    seed_cells: int
    clouds: int


@dataclass(frozen=True)
class RDMethod:
    """Stable paper label and accepted registry aliases for an RD method."""

    key: str
    label: str
    aliases: tuple[str, ...]


RD_METHODS = (
    RDMethod(
        "adam_encoder",
        "Constellation (Adam encoder)",
        ("adam_encoder", "adam_multistart", "adam_ste"),
    ),
    RDMethod("refiner", "Constellation (refiner)", ("refiner", "free")),
    RDMethod("feature_codec", "Feature codec", ("feature_codec", "feature_latent")),
    RDMethod("gpcc", "G-PCC", ("gpcc", "gpcc_octree", "tmc13")),
    RDMethod(
        "draco_subset",
        "Draco on subsets",
        (
            "draco_adam_subset",
            "draco_fps_subset",
            "draco_subset",
            "draco_on_subsets",
        ),
    ),
    RDMethod("pcc_geo_cnn_v2", "pcc_geo_cnn_v2", ("pcc_geo_cnn_v2",)),
    RDMethod("octattention", "OctAttention", ("octattention", "oct_attention")),
    RDMethod("pcgcv2", "PCGCv2", ("pcgcv2", "pcgc_v2")),
)


@dataclass(frozen=True)
class _PreparedPoint:
    label: str
    rates: NDArray[np.float64]
    qualities: NDArray[np.float64]
    decoder_indices: NDArray[np.int64]
    refiner_indices: NDArray[np.int64]
    cloud_indices: NDArray[np.int64]


@dataclass(frozen=True)
class _PreparedCurve:
    points: tuple[_PreparedPoint, ...]
    decoder_levels: tuple[Any, ...]
    refiner_levels: tuple[Any, ...]
    cloud_levels: tuple[tuple[Any, ...], ...]
    experiments: frozenset[str]


def _invalid(
    reason: str,
    *,
    anchor_points: int = 0,
    test_points: int = 0,
) -> BDResult:
    return BDResult(
        value=None,
        reason=reason,
        anchor_points=anchor_points,
        test_points=test_points,
    )


def _arrays(
    rates: Sequence[float] | NDArray[np.floating[Any]],
    qualities: Sequence[float] | NDArray[np.floating[Any]],
    *,
    curve_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str | None]:
    rate_array = np.asarray(rates, dtype=np.float64)
    quality_array = np.asarray(qualities, dtype=np.float64)
    if rate_array.ndim != 1 or quality_array.ndim != 1:
        return rate_array, quality_array, f"{curve_name} curve must be one-dimensional"
    if len(rate_array) != len(quality_array):
        return rate_array, quality_array, f"{curve_name} rate/quality lengths differ"
    if len(rate_array) < MINIMUM_BD_POINTS:
        return (
            rate_array,
            quality_array,
            f"{curve_name} curve has {len(rate_array)} points; at least 4 are required",
        )
    if not np.all(np.isfinite(rate_array)) or not np.all(np.isfinite(quality_array)):
        return (
            rate_array,
            quality_array,
            f"{curve_name} curve contains non-finite values",
        )
    if np.any(rate_array <= 0.0):
        return rate_array, quality_array, f"{curve_name} rates must be positive"
    if len(np.unique(rate_array)) < MINIMUM_BD_POINTS:
        return (
            rate_array,
            quality_array,
            f"{curve_name} curve has fewer than 4 distinct rate values",
        )
    return rate_array, quality_array, None


def _bd_metric(
    anchor_rates: Sequence[float] | NDArray[np.floating[Any]],
    anchor_qualities: Sequence[float] | NDArray[np.floating[Any]],
    test_rates: Sequence[float] | NDArray[np.floating[Any]],
    test_qualities: Sequence[float] | NDArray[np.floating[Any]],
    *,
    mode: str,
) -> BDResult:
    anchor_rate, anchor_quality, reason = _arrays(
        anchor_rates, anchor_qualities, curve_name="anchor"
    )
    test_rate, test_quality, test_reason = _arrays(
        test_rates, test_qualities, curve_name="test"
    )
    anchor_count = len(anchor_rate)
    test_count = len(test_rate)
    if reason is not None:
        return _invalid(reason, anchor_points=anchor_count, test_points=test_count)
    if test_reason is not None:
        return _invalid(test_reason, anchor_points=anchor_count, test_points=test_count)

    if mode == "rate":
        anchor_x, anchor_y = anchor_quality, np.log(anchor_rate)
        test_x, test_y = test_quality, np.log(test_rate)
        domain = "quality"
    elif mode == "psnr":
        anchor_x, anchor_y = np.log(anchor_rate), anchor_quality
        test_x, test_y = np.log(test_rate), test_quality
        domain = "log-rate"
    else:
        raise ValueError(f"unknown BD mode: {mode}")

    if len(np.unique(anchor_x)) < MINIMUM_BD_POINTS:
        return _invalid(
            f"anchor curve has fewer than 4 distinct {domain} values",
            anchor_points=anchor_count,
            test_points=test_count,
        )
    if len(np.unique(test_x)) < MINIMUM_BD_POINTS:
        return _invalid(
            f"test curve has fewer than 4 distinct {domain} values",
            anchor_points=anchor_count,
            test_points=test_count,
        )

    overlap_min = max(float(np.min(anchor_x)), float(np.min(test_x)))
    overlap_max = min(float(np.max(anchor_x)), float(np.max(test_x)))
    if overlap_max <= overlap_min:
        return _invalid(
            f"curves have no positive {domain} overlap",
            anchor_points=anchor_count,
            test_points=test_count,
        )

    try:
        anchor_polynomial = np.polyfit(anchor_x, anchor_y, 3)
        test_polynomial = np.polyfit(test_x, test_y, 3)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _invalid(
            f"cubic interpolation failed: {exc}",
            anchor_points=anchor_count,
            test_points=test_count,
        )
    anchor_integral = np.polyint(anchor_polynomial)
    test_integral = np.polyint(test_polynomial)
    anchor_area = float(
        np.polyval(anchor_integral, overlap_max)
        - np.polyval(anchor_integral, overlap_min)
    )
    test_area = float(
        np.polyval(test_integral, overlap_max) - np.polyval(test_integral, overlap_min)
    )
    average_difference = (test_area - anchor_area) / (overlap_max - overlap_min)
    value = (
        100.0 * math.expm1(average_difference) if mode == "rate" else average_difference
    )
    if not math.isfinite(value):
        return _invalid(
            "cubic interpolation produced a non-finite result",
            anchor_points=anchor_count,
            test_points=test_count,
        )
    display_overlap_min = overlap_min if mode == "rate" else math.exp(overlap_min)
    display_overlap_max = overlap_max if mode == "rate" else math.exp(overlap_max)
    return BDResult(
        value=float(value),
        reason=None,
        overlap_min=display_overlap_min,
        overlap_max=display_overlap_max,
        anchor_points=anchor_count,
        test_points=test_count,
    )


def bd_rate(
    anchor_rates: Sequence[float] | NDArray[np.floating[Any]],
    anchor_psnr: Sequence[float] | NDArray[np.floating[Any]],
    test_rates: Sequence[float] | NDArray[np.floating[Any]],
    test_psnr: Sequence[float] | NDArray[np.floating[Any]],
) -> BDResult:
    """Return cubic log-rate BD-rate (%) for ``test`` relative to ``anchor``."""

    return _bd_metric(
        anchor_rates,
        anchor_psnr,
        test_rates,
        test_psnr,
        mode="rate",
    )


def bd_psnr(
    anchor_rates: Sequence[float] | NDArray[np.floating[Any]],
    anchor_psnr: Sequence[float] | NDArray[np.floating[Any]],
    test_rates: Sequence[float] | NDArray[np.floating[Any]],
    test_psnr: Sequence[float] | NDArray[np.floating[Any]],
) -> BDResult:
    """Return cubic log-rate BD-PSNR (dB) for ``test`` relative to ``anchor``."""

    return _bd_metric(
        anchor_rates,
        anchor_psnr,
        test_rates,
        test_psnr,
        mode="psnr",
    )


def _experiment_rank(name: str) -> tuple[int, str]:
    match = re.search(r"experiment_(\d+)", name)
    return (int(match.group(1)) if match else -1, name)


def _point_label(row: dict[str, Any], rate_field: str) -> str:
    label = row.get("rate_point")
    if label not in {None, "", "unspecified"}:
        return str(label)
    size = row.get("constellation_size")
    bits = row.get("coordinate_bits")
    if size is not None:
        return f"k_{size}_q_{bits}" if bits is not None else f"k_{size}"
    return f"{float(row[rate_field]):.12g}"


def select_curve_rows(
    rows: Sequence[dict[str, Any]],
    aliases: Sequence[str],
    *,
    dataset: str,
    split: str,
    metric_name: str,
    rate_field: str,
) -> list[dict[str, Any]]:
    """Select one measured experiment/arm for a method's registry curve.

    The experiment with the most measured rate points wins; the newer experiment
    breaks ties. This prevents a later single-rate diagnostic from silently
    replacing an earlier measured curve.
    """

    alias_set = set(aliases)
    candidates = [
        row
        for row in rows
        if row.get("record_kind") == "per_cloud"
        and row.get("dataset") == dataset
        and row.get("split") == split
        and row.get("metric_name") == metric_name
        and row.get("method") in alias_set
        and row.get("eligible_for_rd") is not False
        and isinstance(row.get(rate_field), (int, float))
        and not isinstance(row.get(rate_field), bool)
        and math.isfinite(float(row[rate_field]))
        and float(row[rate_field]) > 0.0
    ]
    if not candidates:
        return []
    by_experiment: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        key = (str(row.get("experiment", "")), str(row.get("method", "")))
        by_experiment.setdefault(key, []).append(row)
    variant = max(
        by_experiment,
        key=lambda key: (
            len({_point_label(row, rate_field) for row in by_experiment[key]}),
            _experiment_rank(key[0]),
            -tuple(aliases).index(key[1]),
        ),
    )
    selected = by_experiment[variant]
    arms = {str(row.get("arm_label", "default")) for row in selected}
    for preferred in ("stabilized", "selected", "default"):
        if preferred in arms:
            selected = [row for row in selected if row.get("arm_label") == preferred]
            break
    else:
        arm = max(
            arms,
            key=lambda value: len(
                {
                    _point_label(row, rate_field)
                    for row in selected
                    if str(row.get("arm_label", "default")) == value
                }
            ),
        )
        selected = [row for row in selected if str(row.get("arm_label")) == arm]
    budgets = {
        int(row["budget"])
        for row in selected
        if isinstance(row.get("budget"), int)
        and not isinstance(row.get("budget"), bool)
    }
    if len(budgets) > 1:
        maximum_budget = max(budgets)
        selected = [row for row in selected if row.get("budget") == maximum_budget]
    return selected


def _sort_optional(value: Any) -> tuple[int, str]:
    return (1, "") if value is None else (0, str(value))


def _cloud_key(row: dict[str, Any]) -> tuple[Any, ...]:
    pair_id = row.get("pair_id")
    if pair_id is not None:
        return ("pair_id", pair_id)
    return (
        "cloud",
        row.get("family"),
        row.get("model_id"),
        row.get("sample_id"),
    )


def _prepare_curve(rows: Sequence[dict[str, Any]], rate_field: str) -> _PreparedCurve:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_point_label(row, rate_field), []).append(row)
    if not grouped:
        return _PreparedCurve((), (), (), (), frozenset())

    cloud_sets = [
        {_cloud_key(row) for row in point_rows} for point_rows in grouped.values()
    ]
    common_clouds = set.intersection(*cloud_sets)
    if not common_clouds:
        return _PreparedCurve((), (), (), (), frozenset())
    decoder_levels = tuple(
        sorted({row.get("decoder_seed") for row in rows}, key=_sort_optional)
    )
    refiner_levels = tuple(
        sorted({row.get("refiner_seed") for row in rows}, key=_sort_optional)
    )
    cloud_levels = tuple(sorted(common_clouds, key=lambda value: str(value)))
    decoder_index = {value: index for index, value in enumerate(decoder_levels)}
    refiner_index = {value: index for index, value in enumerate(refiner_levels)}
    cloud_index = {value: index for index, value in enumerate(cloud_levels)}

    prepared_points = []
    for label, point_rows in sorted(grouped.items()):
        cells: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
        for row in point_rows:
            cloud = _cloud_key(row)
            if cloud not in cloud_index:
                continue
            key = (
                decoder_index[row.get("decoder_seed")],
                refiner_index[row.get("refiner_seed")],
                cloud_index[cloud],
            )
            cells.setdefault(key, []).append(
                (float(row[rate_field]), float(row["value"]))
            )
        if not cells:
            continue
        keys = sorted(cells)
        prepared_points.append(
            _PreparedPoint(
                label=label,
                rates=np.asarray(
                    [np.mean([value[0] for value in cells[key]]) for key in keys],
                    dtype=np.float64,
                ),
                qualities=np.asarray(
                    [np.mean([value[1] for value in cells[key]]) for key in keys],
                    dtype=np.float64,
                ),
                decoder_indices=np.asarray([key[0] for key in keys], dtype=np.int64),
                refiner_indices=np.asarray([key[1] for key in keys], dtype=np.int64),
                cloud_indices=np.asarray([key[2] for key in keys], dtype=np.int64),
            )
        )
    return _PreparedCurve(
        points=tuple(prepared_points),
        decoder_levels=decoder_levels,
        refiner_levels=refiner_levels,
        cloud_levels=cloud_levels,
        experiments=frozenset(str(row.get("experiment", "")) for row in rows),
    )


def _factor_counts(rng: np.random.Generator, levels: int) -> NDArray[np.int64]:
    if levels == 1:
        return np.ones(1, dtype=np.int64)
    return np.bincount(rng.integers(0, levels, size=levels), minlength=levels)


def _diagonal_seed_pairing(curve: _PreparedCurve) -> bool:
    """Return whether decoder/refiner labels describe one paired model seed."""

    return (
        len(curve.decoder_levels) > 1
        and curve.decoder_levels == curve.refiner_levels
        and all(
            np.array_equal(point.decoder_indices, point.refiner_indices)
            for point in curve.points
        )
    )


def _curve_seed_counts(
    curve: _PreparedCurve, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    decoder = _factor_counts(rng, len(curve.decoder_levels))
    refiner = (
        decoder.copy()
        if _diagonal_seed_pairing(curve)
        else _factor_counts(rng, len(curve.refiner_levels))
    )
    return decoder, refiner


def _aggregate_prepared(
    curve: _PreparedCurve,
    decoder_counts: NDArray[np.int64],
    refiner_counts: NDArray[np.int64],
    cloud_counts: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rates = []
    qualities = []
    for point in curve.points:
        weights = (
            decoder_counts[point.decoder_indices]
            * refiner_counts[point.refiner_indices]
            * cloud_counts[point.cloud_indices]
        ).astype(np.float64)
        total = float(np.sum(weights))
        if total <= 0.0:
            continue
        rates.append(float(np.dot(weights, point.rates) / total))
        qualities.append(float(np.dot(weights, point.qualities) / total))
    return np.asarray(rates, dtype=np.float64), np.asarray(qualities, dtype=np.float64)


def _unit_counts(curve: _PreparedCurve) -> tuple[NDArray[np.int64], ...]:
    return (
        np.ones(len(curve.decoder_levels), dtype=np.int64),
        np.ones(len(curve.refiner_levels), dtype=np.int64),
        np.ones(len(curve.cloud_levels), dtype=np.int64),
    )


def _dominated(
    rates: NDArray[np.float64], qualities: NDArray[np.float64]
) -> list[bool]:
    result = []
    for index, (rate, quality) in enumerate(zip(rates, qualities, strict=True)):
        dominated = any(
            other_rate <= rate
            and other_quality >= quality
            and (other_rate < rate or other_quality > quality)
            for other_index, (other_rate, other_quality) in enumerate(
                zip(rates, qualities, strict=True)
            )
            if other_index != index
        )
        result.append(dominated)
    return result


def aggregate_rd_curve(
    rows: Sequence[dict[str, Any]],
    *,
    rate_field: str,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260826,
    confidence_level: float = 0.95,
) -> RDCurve:
    """Aggregate measured registry points and bootstrap seeds and paired clouds."""

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    curve = _prepare_curve(rows, rate_field)
    if not curve.points:
        return RDCurve((), 0, 0)
    rates, qualities = _aggregate_prepared(curve, *_unit_counts(curve))
    bootstrap = np.empty((bootstrap_samples, len(curve.points)), dtype=np.float64)
    if bootstrap_samples:
        rng = np.random.default_rng(bootstrap_seed)
        for sample in range(bootstrap_samples):
            decoder_counts, refiner_counts = _curve_seed_counts(curve, rng)
            _, sample_quality = _aggregate_prepared(
                curve,
                decoder_counts,
                refiner_counts,
                _factor_counts(rng, len(curve.cloud_levels)),
            )
            bootstrap[sample] = sample_quality
        alpha = (1.0 - confidence_level) / 2.0
        lower, upper = np.quantile(bootstrap, (alpha, 1.0 - alpha), axis=0)
    else:
        lower = np.full(len(curve.points), np.nan)
        upper = np.full(len(curve.points), np.nan)
    dominated = _dominated(rates, qualities)
    order = np.argsort(rates, kind="stable")
    points = tuple(
        RDCurvePoint(
            rate_point=curve.points[index].label,
            rate=float(rates[index]),
            quality=float(qualities[index]),
            ci_lower=float(lower[index]) if bootstrap_samples else None,
            ci_upper=float(upper[index]) if bootstrap_samples else None,
            dominated=dominated[index],
            observations=len(curve.points[index].rates),
        )
        for index in order
    )
    return RDCurve(
        points=points,
        seed_cells=len(curve.decoder_levels) * len(curve.refiner_levels),
        clouds=len(curve.cloud_levels),
    )


def _restrict_clouds(
    curve: _PreparedCurve, clouds: tuple[tuple[Any, ...], ...]
) -> _PreparedCurve:
    old_indices = {cloud: index for index, cloud in enumerate(curve.cloud_levels)}
    new_indices = {old_indices[cloud]: index for index, cloud in enumerate(clouds)}
    points = []
    for point in curve.points:
        mask = np.asarray(
            [index in new_indices for index in point.cloud_indices], dtype=np.bool_
        )
        points.append(
            replace(
                point,
                rates=point.rates[mask],
                qualities=point.qualities[mask],
                decoder_indices=point.decoder_indices[mask],
                refiner_indices=point.refiner_indices[mask],
                cloud_indices=np.asarray(
                    [new_indices[index] for index in point.cloud_indices[mask]],
                    dtype=np.int64,
                ),
            )
        )
    return replace(curve, points=tuple(points), cloud_levels=clouds)


def _paired_factor_counts(
    anchor: _PreparedCurve,
    test: _PreparedCurve,
    *,
    factor: str,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    anchor_levels = getattr(anchor, f"{factor}_levels")
    test_levels = getattr(test, f"{factor}_levels")
    if (
        anchor.experiments == test.experiments
        and anchor_levels == test_levels
        and len(anchor_levels) > 1
    ):
        counts = _factor_counts(rng, len(anchor_levels))
        return counts, counts.copy()
    return (
        _factor_counts(rng, len(anchor_levels)),
        _factor_counts(rng, len(test_levels)),
    )


def _estimate(
    result: BDResult, values: list[float], samples: int
) -> BDBootstrapEstimate:
    if result.value is None:
        return BDBootstrapEstimate(
            value=None,
            ci_lower=None,
            ci_upper=None,
            reason=result.reason,
            bootstrap_valid=len(values),
            bootstrap_samples=samples,
        )
    required = max(1, math.ceil(samples / 2))
    if len(values) < required:
        return BDBootstrapEstimate(
            value=result.value,
            ci_lower=None,
            ci_upper=None,
            reason=(
                f"only {len(values)} of {samples} bootstrap draws had sufficient "
                "overlap"
            ),
            bootstrap_valid=len(values),
            bootstrap_samples=samples,
            overlap_min=result.overlap_min,
            overlap_max=result.overlap_max,
        )
    lower, upper = np.quantile(np.asarray(values), (0.025, 0.975))
    return BDBootstrapEstimate(
        value=result.value,
        ci_lower=float(lower),
        ci_upper=float(upper),
        reason=None,
        bootstrap_valid=len(values),
        bootstrap_samples=samples,
        overlap_min=result.overlap_min,
        overlap_max=result.overlap_max,
    )


def bootstrap_bd_comparison(
    anchor_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    rate_field: str,
    anchor_rate_field: str | None = None,
    test_rate_field: str | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260826,
) -> BDComparison:
    """Compute paired-cloud BD metrics with decoder/refiner seed uncertainty."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    anchor = _prepare_curve(anchor_rows, anchor_rate_field or rate_field)
    test = _prepare_curve(test_rows, test_rate_field or rate_field)
    common_clouds = tuple(
        sorted(
            set(anchor.cloud_levels) & set(test.cloud_levels),
            key=lambda value: str(value),
        )
    )
    if not anchor.points or not test.points or not common_clouds:
        reason = (
            "no paired clouds"
            if anchor.points and test.points
            else "one or both curves have no measured points"
        )
        unavailable = BDBootstrapEstimate(
            value=None,
            ci_lower=None,
            ci_upper=None,
            reason=reason,
            bootstrap_valid=0,
            bootstrap_samples=bootstrap_samples,
        )
        return BDComparison(unavailable, unavailable, len(common_clouds))
    anchor = _restrict_clouds(anchor, common_clouds)
    test = _restrict_clouds(test, common_clouds)
    anchor_rate, anchor_quality = _aggregate_prepared(anchor, *_unit_counts(anchor))
    test_rate, test_quality = _aggregate_prepared(test, *_unit_counts(test))
    rate_result = bd_rate(anchor_rate, anchor_quality, test_rate, test_quality)
    psnr_result = bd_psnr(anchor_rate, anchor_quality, test_rate, test_quality)

    rng = np.random.default_rng(bootstrap_seed)
    rate_values: list[float] = []
    psnr_values: list[float] = []
    for _ in range(bootstrap_samples):
        anchor_decoder, test_decoder = _paired_factor_counts(
            anchor, test, factor="decoder", rng=rng
        )
        anchor_refiner, test_refiner = _paired_factor_counts(
            anchor, test, factor="refiner", rng=rng
        )
        if _diagonal_seed_pairing(anchor):
            anchor_refiner = anchor_decoder.copy()
        if _diagonal_seed_pairing(test):
            test_refiner = test_decoder.copy()
        cloud_counts = _factor_counts(rng, len(common_clouds))
        anchor_rate_draw, anchor_quality_draw = _aggregate_prepared(
            anchor,
            anchor_decoder,
            anchor_refiner,
            cloud_counts,
        )
        test_rate_draw, test_quality_draw = _aggregate_prepared(
            test,
            test_decoder,
            test_refiner,
            cloud_counts,
        )
        rate_draw = bd_rate(
            anchor_rate_draw,
            anchor_quality_draw,
            test_rate_draw,
            test_quality_draw,
        )
        psnr_draw = bd_psnr(
            anchor_rate_draw,
            anchor_quality_draw,
            test_rate_draw,
            test_quality_draw,
        )
        if rate_draw.value is not None:
            rate_values.append(rate_draw.value)
        if psnr_draw.value is not None:
            psnr_values.append(psnr_draw.value)
    return BDComparison(
        bd_rate=_estimate(rate_result, rate_values, bootstrap_samples),
        bd_psnr=_estimate(psnr_result, psnr_values, bootstrap_samples),
        paired_clouds=len(common_clouds),
    )
