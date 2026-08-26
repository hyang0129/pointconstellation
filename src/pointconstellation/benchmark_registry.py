"""Build and query the machine-readable benchmark results registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REGISTRY_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path("artifacts/local")
DEFAULT_REGISTRY_PATH = DEFAULT_ARTIFACT_ROOT / "benchmark_registry.jsonl"

HEADLINE_METHODS = (
    ("fps", "FPS"),
    ("random_best_of_n", "random best-of-N"),
    ("poisson_disk", "Poisson-disk"),
    ("kmeans", "k-means"),
    ("adam_16", "Adam-16"),
    ("adam_64", "Adam-64"),
    ("adam_256", "Adam-256"),
    ("refiner", "refiner"),
)

_IDENTIFIER_FIELDS = {
    "actual_stream_bpp",
    "actual_bpp",
    "adam_decoder_evaluations",
    "arm",
    "arm_label",
    "bits",
    "bytes",
    "clouds",
    "constellation_size",
    "coordinate_bits",
    "data_seed",
    "dataset",
    "decoder_evaluations",
    "decoder_seed",
    "evaluation_budget",
    "family",
    "feature_bits",
    "input_points",
    "lambda",
    "latent_dim",
    "mean_actual_bpp",
    "mean_stream_bytes",
    "method",
    "model_id",
    "model_seed",
    "nominal_payload_bpp",
    "pair_id",
    "payload_bits",
    "rate_bpp",
    "rate_bytes",
    "rate_label",
    "rate_point",
    "rate_point_bpp",
    "rate_point_bytes",
    "refiner_seed",
    "sample_id",
    "samples",
    "seed",
    "split",
    "stream_bytes",
    "value",
}
_RATE_BYTE_FIELDS = (
    "rate_bytes",
    "rate_point_bytes",
    "stream_bytes",
    "mean_stream_bytes",
    "bytes",
)
_RATE_BPP_FIELDS = (
    "rate_bpp",
    "rate_point_bpp",
    "actual_stream_bpp",
    "mean_actual_bpp",
    "actual_bpp",
)
_OFFICIAL_METRIC_ALIASES = {
    "d1_mse": "official_d1_mse",
    "d1_rmse": "official_d1_rmse",
    "d1_psnr_db": "official_d1_psnr_db",
    "d1_hausdorff": "official_d1_hausdorff",
    "d1_hausdorff_psnr_db": "official_d1_hausdorff_psnr_db",
    "d2_mse": "official_d2_mse",
    "d2_rmse": "official_d2_rmse",
    "d2_psnr_db": "official_d2_psnr_db",
    "d2_hausdorff": "official_d2_hausdorff",
    "d2_hausdorff_psnr_db": "official_d2_hausdorff_psnr_db",
    "official_d1_rmse_grid_units": "official_d1_rmse",
    "official_d2_rmse_grid_units": "official_d2_rmse",
}


@dataclass(frozen=True)
class BenchmarkRegistryConfig:
    """Paths used by a deterministic registry rebuild."""

    artifact_root: str = str(DEFAULT_ARTIFACT_ROOT)
    output_path: str = str(DEFAULT_REGISTRY_PATH)

    @classmethod
    def from_json(cls, path: Path) -> BenchmarkRegistryConfig:
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class _SourceContext:
    dataset: str
    experiment: str
    input_points: int | None = None
    model_seed: int | None = None
    default_arm: str = "default"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of an artifact without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark artifact {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    _line_number = 0
    try:
        with path.open() as handle:
            for _line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"cannot read benchmark artifact {path} at line {_line_number}: {exc}"
        ) from exc
    return rows


def _nested(document: dict[str, Any], *keys: str) -> Any:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_integer(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _experiment_name(path: Path) -> str:
    for part in reversed(path.parts):
        if re.match(r"experiment_\d+", part):
            return part
    return path.parent.name


def _context_from_document(
    document: dict[str, Any],
    path: Path,
    *,
    fallback_dataset: str | None = None,
    default_arm: str = "default",
) -> _SourceContext:
    dataset_candidates = (
        _nested(document, "protocol", "data_identity", "dataset"),
        _nested(document, "protocol", "dataset"),
        _nested(document, "data_protocol", "dataset"),
        document.get("dataset"),
        _nested(document, "config", "dataset"),
    )
    dataset = next(
        (
            str(candidate)
            for candidate in dataset_candidates
            if isinstance(candidate, str) and candidate
        ),
        fallback_dataset,
    )
    if dataset is None:
        dataset = "ModelNet40" if "modelnet40" in str(path).lower() else "unknown"
    input_points = _first_integer(
        _nested(document, "protocol", "input_points"),
        _nested(document, "config", "num_points"),
        _nested(document, "config", "experiment", "num_points"),
        document.get("input_points"),
    )
    model_seed = _first_integer(
        document.get("model_seed"),
        document.get("decoder_seed"),
        _nested(document, "config", "model_seed"),
        _nested(document, "config", "decoder_seed"),
        _nested(document, "config", "experiment", "seed"),
    )
    arm = next(
        (
            str(candidate)
            for candidate in (
                document.get("arm_label"),
                document.get("arm"),
                _nested(document, "config", "decoder_arm"),
                default_arm,
            )
            if isinstance(candidate, str) and candidate
        ),
        default_arm,
    )
    return _SourceContext(
        dataset=dataset,
        experiment=_experiment_name(path),
        input_points=input_points,
        model_seed=model_seed,
        default_arm=arm,
    )


def _number(row: dict[str, Any], fields: Iterable[str]) -> float | None:
    for field in fields:
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"non-finite {field} in benchmark row")
            return result
    return None


def canonical_method(method: str, row: dict[str, Any] | None = None) -> str:
    """Map producer-specific method labels onto stable registry labels."""

    normalized = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
    aliases = {
        "farthest_point_sampling": "fps",
        "free": "refiner",
        "competitive_refiner": "refiner",
        "random_best_n": "random_best_of_n",
        "best_of_n_random": "random_best_of_n",
        "best_of_random": "random_best_of_n",
        "poisson": "poisson_disk",
        "poisson_disk_sampling": "poisson_disk",
        "k_means": "kmeans",
    }
    normalized = aliases.get(normalized, normalized)
    if re.fullmatch(r"random_best_of_(?:n|\d+)", normalized) or re.fullmatch(
        r"best_of_\d+_random", normalized
    ):
        return "random_best_of_n"
    if normalized in {"adam", "adam_probe", "adam_search"}:
        values = row or {}
        budget = _first_integer(
            values.get("adam_decoder_evaluations"),
            values.get("decoder_evaluations"),
            values.get("evaluation_budget"),
        )
        if budget is not None:
            return f"adam_{budget}"
    match = re.fullmatch(r"adam_(?:probe_)?(\d+)", normalized)
    if match:
        return f"adam_{int(match.group(1))}"
    return normalized


def _metric_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return _OFFICIAL_METRIC_ALIASES.get(normalized, normalized)


def _metric_values(row: dict[str, Any]) -> list[tuple[str, float]]:
    explicit_name = row.get("metric_name", row.get("metric"))
    explicit_value = row.get("value")
    if isinstance(explicit_name, str) and isinstance(explicit_value, (int, float)):
        if isinstance(explicit_value, bool) or not math.isfinite(float(explicit_value)):
            raise ValueError("registry input contains a non-finite metric value")
        return [(_metric_name(explicit_name), float(explicit_value))]

    values: list[tuple[str, float]] = []
    for name, value in row.items():
        if name in {"metrics", "official_metrics"} and isinstance(value, dict):
            for nested_name, nested_value in value.items():
                if isinstance(nested_value, (int, float)) and not isinstance(
                    nested_value, bool
                ):
                    number = float(nested_value)
                    if not math.isfinite(number):
                        raise ValueError("registry input contains a non-finite metric")
                    values.append((_metric_name(nested_name), number))
            continue
        if name in _IDENTIFIER_FIELDS or not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("registry input contains a non-finite metric")
        values.append((_metric_name(name), number))
    return sorted(set(values))


def _rate(row: dict[str, Any], context: _SourceContext) -> tuple[float | None, ...]:
    rate_bytes = _number(row, _RATE_BYTE_FIELDS)
    rate_bpp = _number(row, _RATE_BPP_FIELDS)
    if rate_bytes is None and rate_bpp is not None and context.input_points:
        rate_bytes = rate_bpp * context.input_points / 8.0
    if rate_bpp is None and rate_bytes is not None and context.input_points:
        rate_bpp = 8.0 * rate_bytes / context.input_points
    return rate_bytes, rate_bpp


def _rate_label(row: dict[str, Any], rate_bytes: float | None) -> str:
    for field in ("rate_point", "rate_label", "lambda"):
        if row.get(field) is not None:
            return str(row[field])
    if row.get("constellation_size") is not None:
        return f"k_{int(row['constellation_size'])}"
    if rate_bytes is not None:
        return f"{rate_bytes:g}B"
    return "unspecified"


def _seed_fields(
    row: dict[str, Any], context: _SourceContext, *, raw_method: str
) -> tuple[int | None, int | None]:
    method = canonical_method(raw_method, row)
    decoder_seed = _first_integer(row.get("decoder_seed"), row.get("model_seed"))
    refiner_seed = _first_integer(row.get("refiner_seed"))
    if decoder_seed is None and method not in {"gpcc", "gpcc_octree"}:
        decoder_seed = context.model_seed
    if (
        refiner_seed is None
        and context.experiment.startswith("experiment_017")
        and method in {"refiner", "strict_subset"}
    ):
        refiner_seed = context.model_seed
    return decoder_seed, refiner_seed


def _normalize_observation(
    observation: dict[str, Any],
    *,
    context: _SourceContext,
    source_path: Path,
    source_sha256: str,
) -> list[dict[str, Any]]:
    raw_method = observation.get("method")
    split = observation.get("split")
    if not isinstance(raw_method, str) or not isinstance(split, str):
        return []
    method = canonical_method(raw_method, observation)
    rate_bytes, rate_bpp = _rate(observation, context)
    decoder_seed, refiner_seed = _seed_fields(
        observation, context, raw_method=raw_method
    )
    arm = observation.get("arm_label", observation.get("arm"))
    if not isinstance(arm, str) or not arm:
        arm = context.default_arm
    dataset = observation.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        dataset = context.dataset
    base = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "method": method,
        "arm_label": str(arm),
        "rate_point": _rate_label(observation, rate_bytes),
        "rate_bytes": rate_bytes,
        "rate_bpp": rate_bpp,
        # Familiar aliases make the actual-rate definition explicit to readers.
        "stream_bytes": rate_bytes,
        "actual_stream_bpp": rate_bpp,
        "decoder_seed": decoder_seed,
        "refiner_seed": refiner_seed,
        "sample_id": observation.get("sample_id"),
        "family": observation.get("family"),
        "model_id": observation.get("model_id"),
        "constellation_size": observation.get("constellation_size"),
        "coordinate_bits": observation.get("coordinate_bits"),
        "record_kind": "per_cloud",
        "experiment": context.experiment,
        "source_path": source_path.as_posix(),
        "source_sha256": source_sha256,
    }
    return [
        {**base, "metric_name": name, "value": value}
        for name, value in _metric_values(observation)
    ]


def _rows_from_source(
    observations: Iterable[dict[str, Any]],
    *,
    context: _SourceContext,
    source_path: Path,
) -> list[dict[str, Any]]:
    digest = file_sha256(source_path)
    rows = []
    for observation in observations:
        if isinstance(observation, dict):
            rows.extend(
                _normalize_observation(
                    observation,
                    context=context,
                    source_path=source_path,
                    source_sha256=digest,
                )
            )
    return rows


def _embedded_per_cloud(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    direct = document.get("per_cloud")
    if isinstance(direct, list):
        rows.extend(direct)
    for section_name in ("gpcc", "feature_codec"):
        section = document.get(section_name)
        if isinstance(section, dict) and isinstance(section.get("per_cloud"), list):
            rows.extend(section["per_cloud"])
    return rows


def _import_experiment_017(root: Path) -> list[dict[str, Any]]:
    rows = []
    experiment = root / "experiment_017_modelnet40_multiseed"
    for path in sorted(experiment.glob("*/benchmark_metrics.json")):
        document = _read_json(path)
        if not isinstance(document, dict):
            raise ValueError(f"benchmark metrics must be a JSON object: {path}")
        context = _context_from_document(document, path)
        rows.extend(
            _rows_from_source(
                _embedded_per_cloud(document), context=context, source_path=path
            )
        )
    return rows


def _import_experiment_019(root: Path) -> tuple[list[dict[str, Any]], str | None]:
    experiment = root / "experiment_019_stability_modelnet40"
    metrics_path = experiment / "stability_metrics.json"
    document: dict[str, Any] = {}
    if metrics_path.is_file():
        value = _read_json(metrics_path)
        if not isinstance(value, dict):
            raise ValueError(f"stability metrics must be a JSON object: {metrics_path}")
        document = value
    context = _context_from_document(document, metrics_path)
    rows_path = experiment / "per_cloud.jsonl"
    if rows_path.is_file():
        rows = _rows_from_source(
            _read_jsonl(rows_path), context=context, source_path=rows_path
        )
    elif metrics_path.is_file():
        rows = _rows_from_source(
            _embedded_per_cloud(document), context=context, source_path=metrics_path
        )
    else:
        rows = []
    return rows, context.dataset if document else None


def _import_experiment_020(
    root: Path, *, fallback_dataset: str | None
) -> list[dict[str, Any]]:
    experiment = root / "experiment_020_official_stability"
    metrics_path = experiment / "official_metrics.json"
    document: dict[str, Any] = {}
    if metrics_path.is_file():
        value = _read_json(metrics_path)
        if not isinstance(value, dict):
            raise ValueError(f"official metrics must be a JSON object: {metrics_path}")
        document = value
    context = _context_from_document(
        document,
        metrics_path,
        fallback_dataset=fallback_dataset,
        default_arm="stabilized",
    )
    rows_path = experiment / "official_per_cloud.jsonl"
    if not rows_path.is_file():
        return []
    observations = _read_jsonl(rows_path)
    declared_rows = document.get("per_cloud_rows")
    if isinstance(declared_rows, int) and declared_rows != len(observations):
        raise ValueError(
            "official_metrics.json per_cloud_rows does not match "
            "official_per_cloud.jsonl"
        )
    return _rows_from_source(observations, context=context, source_path=rows_path)


def _find_context_document(directory: Path) -> tuple[dict[str, Any], Path | None]:
    for path in sorted(directory.glob("*metrics.json")):
        value = _read_json(path)
        if isinstance(value, dict):
            return value, path
    return {}, None


def _generic_per_cloud(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [row for row in document if isinstance(row, dict)]
    if not isinstance(document, dict):
        return []
    rows = _embedded_per_cloud(document)
    for key in ("rows", "official_per_cloud", "results"):
        value = document.get(key)
        if (
            isinstance(value, list)
            and all(isinstance(row, dict) for row in value)
            and any("method" in row and "split" in row for row in value)
        ):
            rows.extend(value)
    return rows


def _import_future_experiments(root: Path) -> list[dict[str, Any]]:
    rows = []
    directories = sorted(
        path
        for pattern in ("experiment_021*", "experiment_022*")
        for path in root.glob(pattern)
        if path.is_dir()
    )
    for directory in directories:
        document, document_path = _find_context_document(directory)
        context_path = document_path or directory
        context = _context_from_document(document, context_path)
        jsonl_paths = sorted(directory.rglob("*per_cloud*.jsonl"))
        for path in jsonl_paths:
            rows.extend(
                _rows_from_source(_read_jsonl(path), context=context, source_path=path)
            )
        if not jsonl_paths:
            for path in sorted(directory.rglob("*metrics.json")):
                value = document if path == document_path else _read_json(path)
                observations = _generic_per_cloud(value)
                if observations:
                    local_context = _context_from_document(
                        value if isinstance(value, dict) else document,
                        path,
                        fallback_dataset=context.dataset,
                    )
                    rows.extend(
                        _rows_from_source(
                            observations, context=local_context, source_path=path
                        )
                    )
    return rows


def _optional_sort(value: Any) -> tuple[int, Any]:
    return (1, "") if value is None else (0, value)


def _row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    fields = (
        "dataset",
        "split",
        "method",
        "arm_label",
        "rate_point",
        "rate_bytes",
        "rate_bpp",
        "decoder_seed",
        "refiner_seed",
        "sample_id",
        "family",
        "model_id",
        "metric_name",
        "record_kind",
        "experiment",
    )
    return tuple(row.get(field) for field in fields)


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _optional_sort(row.get(field))
        for field in (
            "dataset",
            "split",
            "method",
            "arm_label",
            "rate_bytes",
            "rate_bpp",
            "decoder_seed",
            "refiner_seed",
            "family",
            "model_id",
            "sample_id",
            "metric_name",
            "experiment",
            "source_path",
        )
    )


def collect_registry_rows(artifact_root: Path) -> list[dict[str, Any]]:
    """Read supported experiment artifacts and return stable long-form rows."""

    rows = _import_experiment_017(artifact_root)
    stability_rows, stability_dataset = _import_experiment_019(artifact_root)
    rows.extend(stability_rows)
    rows.extend(
        _import_experiment_020(artifact_root, fallback_dataset=stability_dataset)
    )
    rows.extend(_import_future_experiments(artifact_root))

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=_sort_key):
        identity = _row_identity(row)
        previous = unique.get(identity)
        if previous is not None and previous["value"] != row["value"]:
            raise ValueError(
                "conflicting benchmark values for the same registry identity: "
                f"{identity}"
            )
        unique.setdefault(identity, row)
    return sorted(unique.values(), key=_sort_key)


def build_registry(
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_path: Path = DEFAULT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    """Rebuild ``output_path`` atomically and return its deterministic rows."""

    rows = collect_registry_rows(artifact_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(output_path)
    return rows


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> list[dict[str, Any]]:
    """Load a registry and check its schema version and row shape."""

    rows = _read_jsonl(path)
    required = {
        "schema_version",
        "dataset",
        "split",
        "method",
        "arm_label",
        "rate_bytes",
        "rate_bpp",
        "decoder_seed",
        "refiner_seed",
        "metric_name",
        "value",
        "source_path",
        "source_sha256",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"registry row {index} is missing {sorted(missing)}")
        if row["schema_version"] != REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported registry schema in row {index}")
    return rows


def _experiment_rank(name: str) -> tuple[int, str]:
    match = re.search(r"experiment_(\d+)", name)
    return (int(match.group(1)) if match else -1, name)


def _cloud_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("decoder_seed"),
        row.get("family"),
        row.get("model_id"),
        row.get("sample_id"),
    )


def _method_values(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = float(row["value"])
        if row["metric_name"] == "official_d1_rmse":
            value *= value
        grouped.setdefault(_cloud_key(row), []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _choose_method_rows(
    rows: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    selected = [row for row in rows if canonical_method(row["method"], row) == method]
    if not selected:
        return []
    experiments = {str(row.get("experiment", "")) for row in selected}
    experiment = max(experiments, key=_experiment_rank)
    selected = [row for row in selected if row.get("experiment", "") == experiment]
    arms = {str(row.get("arm_label", "default")) for row in selected}
    for preferred in ("stabilized", "selected", "default"):
        if preferred in arms:
            return [row for row in selected if row.get("arm_label") == preferred]
    return selected


def _select_dataset(rows: list[dict[str, Any]], dataset: str | None) -> str | None:
    available = sorted({str(row["dataset"]) for row in rows})
    if dataset is not None:
        if dataset not in available:
            raise ValueError(f"dataset {dataset!r} is absent from the registry")
        return dataset
    for candidate in available:
        if candidate.lower() == "modelnet40":
            return candidate
    return available[0] if available else None


def _select_split(rows: list[dict[str, Any]], panel: str) -> str | None:
    available = {str(row["split"]) for row in rows}
    preferences = ("validation",) if panel == "validation" else ("ood", "category_ood")
    return next((split for split in preferences if split in available), None)


def headline_statistics(
    rows: list[dict[str, Any]],
    *,
    rate_bytes: float = 50.0,
    dataset: str | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260824,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute paired-bootstrap official D1 RMSE for Figure 1 and Table 1."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    candidates = [
        row
        for row in rows
        if row.get("record_kind") == "per_cloud"
        and row.get("metric_name") in {"official_d1_mse", "official_d1_rmse"}
        and row.get("rate_bytes") is not None
        and math.isclose(float(row["rate_bytes"]), rate_bytes, abs_tol=1e-9)
    ]
    selected_dataset = _select_dataset(candidates, dataset)
    if selected_dataset is not None:
        candidates = [row for row in candidates if row["dataset"] == selected_dataset]
    result: dict[str, Any] = {
        "dataset": selected_dataset,
        "rate_bytes": rate_bytes,
        "confidence_level": confidence_level,
        "panels": {},
    }
    alpha = (1.0 - confidence_level) / 2.0
    for panel_index, panel in enumerate(("validation", "ood")):
        source_split = _select_split(candidates, panel)
        split_rows = (
            [row for row in candidates if row["split"] == source_split]
            if source_split is not None
            else []
        )
        method_values = {
            method: _method_values(_choose_method_rows(split_rows, method))
            for method, _ in HEADLINE_METHODS
        }
        method_values = {
            method: values for method, values in method_values.items() if values
        }
        common_keys = (
            sorted(
                set.intersection(*(set(values) for values in method_values.values()))
            )
            if method_values
            else []
        )
        paired_draws = None
        if common_keys:
            rng = np.random.default_rng(bootstrap_seed + 10_000 * panel_index)
            paired_draws = rng.integers(
                0,
                len(common_keys),
                size=(bootstrap_samples, len(common_keys)),
            )
        statistics = {}
        for method_index, (method, label) in enumerate(HEADLINE_METHODS):
            values_by_key = method_values.get(method)
            if not values_by_key:
                continue
            keys = common_keys or sorted(values_by_key)
            values = np.asarray([values_by_key[key] for key in keys], dtype=np.float64)
            point = math.sqrt(float(values.mean()))
            if paired_draws is None:
                rng = np.random.default_rng(
                    bootstrap_seed + 10_000 * panel_index + method_index
                )
                draws = rng.integers(
                    0, len(values), size=(bootstrap_samples, len(values))
                )
            else:
                draws = paired_draws
            bootstrap = np.sqrt(values[draws].mean(axis=1))
            lower, upper = np.quantile(bootstrap, (alpha, 1.0 - alpha))
            statistics[method] = {
                "label": label,
                "rmse": point,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "paired_units": len(values),
            }
        result["panels"][panel] = {
            "source_split": source_split,
            "methods": statistics,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the registry from source artifacts",
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("pass --rebuild to regenerate the registry")
    rows = build_registry(args.artifact_root, args.output)
    print(json.dumps({"output": str(args.output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
