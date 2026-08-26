"""Create immutable mesh manifests for external surface benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from pointconstellation.data.mesh import (
    file_sha256,
    filter_degenerate_faces,
    load_mesh,
)
from pointconstellation.data.pointcloud import load_pointcloud_container

MODELNET40_URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"
THINGI10K_URL = "https://github.com/Thingi10K/Thingi10K"
SCANOBJECTNN_URL = "https://github.com/hkust-vgd/scanobjectnn"
BENCHMARK_SPLITS = ("train", "calibration", "validation", "ood", "final")


def _rank(seed: int, category: str, model_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{model_id}".encode()).hexdigest()


def discover_shapenet_meshes(root: Path) -> dict[str, list[dict[str, str]]]:
    """Discover the ShapeNetCore v2 ``model_normalized.obj`` layout."""

    root = root.resolve()
    discovered: dict[str, list[dict[str, str]]] = {}
    for path in sorted(root.glob("*/*/models/model_normalized.obj")):
        relative = path.relative_to(root)
        category, model_id = relative.parts[:2]
        discovered.setdefault(category, []).append(
            {
                "category": category,
                "model_id": model_id,
                "mesh": relative.as_posix(),
                "mesh_sha256": file_sha256(path),
            }
        )
    if not discovered:
        raise FileNotFoundError(
            "no ShapeNetCore v2 meshes found under "
            f"{root}; expected <synset>/<model>/models/model_normalized.obj"
        )
    return discovered


def create_pilot_manifest(
    root: Path,
    *,
    train_categories: tuple[str, ...],
    heldout_categories: tuple[str, ...],
    train_per_category: int,
    validation_per_category: int,
    category_ood_per_category: int,
    seed: int,
) -> dict[str, Any]:
    """Select deterministic, disjoint model IDs for the external pilot."""

    if not train_categories or not heldout_categories:
        raise ValueError("train and held-out categories must be nonempty")
    if set(train_categories) & set(heldout_categories):
        raise ValueError("train and held-out categories must be disjoint")
    if min(train_per_category, validation_per_category, category_ood_per_category) < 1:
        raise ValueError("per-category sample counts must be positive")

    discovered = discover_shapenet_meshes(root)
    requested = set(train_categories) | set(heldout_categories)
    missing = requested - discovered.keys()
    if missing:
        raise ValueError(f"requested categories are absent: {sorted(missing)}")

    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "category_ood": [],
    }
    for category in train_categories:
        records = sorted(
            discovered[category],
            key=lambda record: _rank(seed, category, record["model_id"]),
        )
        required = train_per_category + validation_per_category
        if len(records) < required:
            raise ValueError(
                f"category {category} has {len(records)} meshes; need {required}"
            )
        splits["train"].extend(records[:train_per_category])
        splits["validation"].extend(records[train_per_category:required])
    for category in heldout_categories:
        records = sorted(
            discovered[category],
            key=lambda record: _rank(seed, category, record["model_id"]),
        )
        if len(records) < category_ood_per_category:
            raise ValueError(
                f"category {category} has {len(records)} meshes; "
                f"need {category_ood_per_category}"
            )
        splits["category_ood"].extend(records[:category_ood_per_category])

    for records in splits.values():
        records.sort(key=lambda record: (record["category"], record["model_id"]))
    return {
        "version": 1,
        "dataset": "ShapeNetCore.v2",
        "seed": seed,
        "sampling": {
            "source_target": "independent_area_weighted_mesh_surface_samples",
            "normalization": "mesh_bbox_center_then_unit_max_vertex_radius",
        },
        "categories": {
            "train": list(train_categories),
            "heldout": list(heldout_categories),
        },
        "splits": splits,
    }


def discover_modelnet40_meshes(
    root: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Discover the official ``<class>/<train|test>/*.off`` layout."""

    root = root.resolve()
    discovered: dict[str, dict[str, list[dict[str, str]]]] = {
        "train": {},
        "test": {},
    }
    for official_split in discovered:
        for path in sorted(root.glob(f"*/{official_split}/*.off")):
            relative = path.relative_to(root)
            category = relative.parts[0]
            discovered[official_split].setdefault(category, []).append(
                {
                    "category": category,
                    "model_id": path.stem,
                    "mesh": relative.as_posix(),
                    "mesh_sha256": file_sha256(path),
                    "official_split": official_split,
                }
            )
    train_categories = set(discovered["train"])
    test_categories = set(discovered["test"])
    if not train_categories or not test_categories:
        raise FileNotFoundError(
            "no ModelNet40 meshes found under "
            f"{root}; expected <class>/<train|test>/*.off"
        )
    if train_categories != test_categories:
        raise ValueError("ModelNet40 train/test category sets do not match")
    return discovered


def split_modelnet40_categories(
    categories: tuple[str, ...], *, heldout_count: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create a deterministic category-level train/OOD partition."""

    unique = tuple(sorted(set(categories)))
    if len(unique) != len(categories):
        raise ValueError("ModelNet40 categories must be unique")
    if not 1 <= heldout_count < len(unique):
        raise ValueError("heldout_count must leave train and held-out categories")
    ranked = sorted(
        unique,
        key=lambda category: _rank(seed, category, "heldout_category"),
    )
    heldout = tuple(sorted(ranked[:heldout_count]))
    train = tuple(sorted(set(unique) - set(heldout)))
    return train, heldout


def create_modelnet40_manifest(
    root: Path,
    *,
    train_categories: tuple[str, ...],
    heldout_categories: tuple[str, ...],
    train_per_category: int,
    calibration_per_category: int = 0,
    validation_per_category: int,
    category_ood_per_category: int,
    seed: int,
    archive_sha256: str | None = None,
    source_url: str = MODELNET40_URL,
) -> dict[str, Any]:
    """Select deterministic subsets while preserving official split membership."""

    if not train_categories or not heldout_categories:
        raise ValueError("train and held-out categories must be nonempty")
    if set(train_categories) & set(heldout_categories):
        raise ValueError("train and held-out categories must be disjoint")
    if min(train_per_category, validation_per_category, category_ood_per_category) < 1:
        raise ValueError("per-category sample counts must be positive")
    if calibration_per_category < 0:
        raise ValueError("calibration_per_category cannot be negative")

    discovered = discover_modelnet40_meshes(root)
    available = set(discovered["train"])
    requested = set(train_categories) | set(heldout_categories)
    missing = requested - available
    if missing:
        raise ValueError(f"requested categories are absent: {sorted(missing)}")

    splits: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "category_ood": [],
    }
    if calibration_per_category:
        splits["calibration"] = []
    for category in train_categories:
        train_records = sorted(
            discovered["train"][category],
            key=lambda record: _rank(seed, category, record["model_id"]),
        )
        test_records = sorted(
            discovered["test"][category],
            key=lambda record: _rank(seed, category, record["model_id"]),
        )
        required_train = train_per_category + calibration_per_category
        if len(train_records) < required_train:
            raise ValueError(
                f"category {category} has {len(train_records)} train meshes; "
                f"need {required_train}"
            )
        if len(test_records) < validation_per_category:
            raise ValueError(
                f"category {category} has {len(test_records)} test meshes; "
                f"need {validation_per_category}"
            )
        splits["train"].extend(train_records[:train_per_category])
        if calibration_per_category:
            splits["calibration"].extend(
                train_records[train_per_category:required_train]
            )
        splits["validation"].extend(test_records[:validation_per_category])
    for category in heldout_categories:
        test_records = sorted(
            discovered["test"][category],
            key=lambda record: _rank(seed, category, record["model_id"]),
        )
        if len(test_records) < category_ood_per_category:
            raise ValueError(
                f"category {category} has {len(test_records)} test meshes; "
                f"need {category_ood_per_category}"
            )
        splits["category_ood"].extend(test_records[:category_ood_per_category])

    for records in splits.values():
        records.sort(key=lambda record: (record["category"], record["model_id"]))
    source: dict[str, str] = {"url": source_url}
    if archive_sha256 is not None:
        source["archive_sha256"] = archive_sha256
    return {
        "version": 1,
        "dataset": "ModelNet40",
        "seed": seed,
        "source": source,
        "sampling": {
            "source_target": "independent_area_weighted_mesh_surface_samples",
            "normalization": "mesh_bbox_center_then_unit_max_vertex_radius",
            "official_split_policy": (
                "train and calibration use disjoint official train meshes; "
                "validation and category_ood use official test"
                if calibration_per_category
                else "train uses official train; validation and category_ood use "
                "official test"
            ),
        },
        "categories": {
            "train": list(train_categories),
            "heldout": list(heldout_categories),
        },
        "splits": splits,
    }


def _stratified_take(
    records: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    namespace: str,
    stratum_fields: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select an exact proportional allocation with deterministic tie breaks."""

    if not 0 <= count <= len(records):
        raise ValueError(f"cannot select {count} records from {len(records)}")
    if count == 0:
        return [], list(records)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in records:
        stratum = tuple(str(record[field]) for field in stratum_fields)
        groups.setdefault(stratum, []).append(record)
    for stratum, members in groups.items():
        members.sort(
            key=lambda record: _rank(
                seed,
                namespace + ":" + ":".join(stratum),
                str(record["model_id"]),
            )
        )

    total = len(records)
    ideals = {
        stratum: count * len(members) / total for stratum, members in groups.items()
    }
    allocation = {
        stratum: min(len(groups[stratum]), int(ideal))
        for stratum, ideal in ideals.items()
    }
    remaining_slots = count - sum(allocation.values())
    ranked_strata = sorted(
        groups,
        key=lambda stratum: (
            -(ideals[stratum] - allocation[stratum]),
            hashlib.sha256(
                f"{seed}:{namespace}:{':'.join(stratum)}".encode()
            ).hexdigest(),
        ),
    )
    while remaining_slots:
        progressed = False
        for stratum in ranked_strata:
            if allocation[stratum] < len(groups[stratum]):
                allocation[stratum] += 1
                remaining_slots -= 1
                progressed = True
                if not remaining_slots:
                    break
        if not progressed:  # pragma: no cover - guarded by the initial count check
            raise RuntimeError("stratified allocation could not fill its quota")

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for stratum, members in groups.items():
        taken = members[: allocation[stratum]]
        selected.extend(taken)
        selected_ids.update(str(record["model_id"]) for record in taken)
    remainder = [
        record for record in records if str(record["model_id"]) not in selected_ids
    ]
    selected.sort(key=lambda record: (str(record["category"]), str(record["model_id"])))
    return selected, remainder


def _partition_records(
    records: list[dict[str, Any]],
    split_counts: dict[str, int],
    *,
    seed: int,
    stratum_fields: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    if any(count < 1 for count in split_counts.values()):
        raise ValueError("benchmark split counts must be positive")
    if sum(split_counts.values()) > len(records):
        raise ValueError(
            f"need {sum(split_counts.values())} records but only {len(records)} exist"
        )
    remaining = list(records)
    splits: dict[str, list[dict[str, Any]]] = {}
    # Select small sealed/evaluation roles before training so their strata do
    # not inherit only the tail left by the much larger training allocation.
    order = sorted(split_counts, key=lambda split: (split_counts[split], split))
    for split in order:
        selected, remaining = _stratified_take(
            remaining,
            split_counts[split],
            seed=seed,
            namespace=split,
            stratum_fields=stratum_fields,
        )
        splits[split] = [{**record, "manifest_role": split} for record in selected]
    return {split: splits[split] for split in split_counts}


def _normalized_field(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _metadata_value(record: dict[str, Any], *names: str) -> Any:
    normalized = {_normalized_field(str(key)): value for key, value in record.items()}
    for name in names:
        key = _normalized_field(name)
        if key in normalized:
            value = normalized[key]
            if value is not None and value != "":
                return value
    return None


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as handle:
            return [dict(record) for record in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list):
            raise ValueError("Thingi10K JSON metadata must contain a list")
        return value
    raise ValueError("Thingi10K metadata must be CSV, JSON, or JSONL")


def _integer_metadata(record: dict[str, Any], *names: str) -> int | None:
    value = _metadata_value(record, *names)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer metadata value: {value!r}") from exc


def _boolean_metadata(record: dict[str, Any], *names: str) -> bool | None:
    value = _metadata_value(record, *names)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean metadata value: {value!r}")


def _thingi10k_size_proxy(face_count: int) -> str:
    if face_count < 2_000:
        return "faces_500_1999"
    if face_count < 8_000:
        return "faces_2000_7999"
    return "faces_8000_plus"


def _thingi10k_genus_proxy(metadata: dict[str, Any]) -> str:
    closed = _boolean_metadata(metadata, "closed")
    components = _integer_metadata(
        metadata, "num_components", "num_connected_components"
    )
    euler = _integer_metadata(metadata, "euler", "euler_characteristic")
    if closed is not True or components is None or euler is None:
        return "open_or_unknown"
    estimate = max(0.0, (2.0 * components - euler) / 2.0)
    if estimate < 0.5:
        return "closed_genus_0"
    if estimate < 1.5:
        return "closed_genus_1"
    return "closed_genus_2_plus"


def discover_thingi10k_meshes(
    root: Path, metadata_path: Path, *, minimum_faces: int = 500
) -> list[dict[str, Any]]:
    """Discover supported Thingi10K files and attach audited local metadata."""

    if minimum_faces < 1:
        raise ValueError("minimum_faces must be positive")
    root = root.resolve()
    metadata_records = _read_metadata(metadata_path)
    by_id: dict[str, dict[str, Any]] = {}
    for metadata in metadata_records:
        file_id = _metadata_value(metadata, "file_id", "id")
        if file_id is not None:
            by_id[str(file_id)] = metadata
    if not by_id:
        raise ValueError("Thingi10K metadata contains no file IDs")

    discovered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".obj", ".off", ".stl"}
    ):
        file_id = path.stem
        metadata = by_id.get(file_id)
        if metadata is None:
            continue
        if file_id in seen_ids:
            raise ValueError(f"duplicate Thingi10K file ID under root: {file_id}")
        seen_ids.add(file_id)
        try:
            mesh = filter_degenerate_faces(load_mesh(path))
        except (OSError, UnicodeError, ValueError):
            # The official dataset deliberately retains documented corrupt and
            # empty inputs. They are excluded by this declared validity rule.
            continue
        face_count = len(mesh.faces)
        if face_count < minimum_faces:
            continue
        license_name = _metadata_value(metadata, "license")
        if license_name is None:
            license_name = "unknown"
        category = _metadata_value(metadata, "category")
        if category is None:
            category = "unknown"
        relative = path.relative_to(root)
        size_proxy = _thingi10k_size_proxy(face_count)
        genus_proxy = _thingi10k_genus_proxy(metadata)
        discovered.append(
            {
                "category": str(category),
                "model_id": file_id,
                "mesh": relative.as_posix(),
                "mesh_sha256": file_sha256(path),
                "license": str(license_name),
                "valid_face_count": face_count,
                "size_proxy": size_proxy,
                "genus_proxy": genus_proxy,
                "stratum": f"{size_proxy}/{genus_proxy}",
            }
        )
    if not discovered:
        raise FileNotFoundError(
            "no valid Thingi10K OBJ/OFF/STL meshes matched the metadata and face filter"
        )
    return discovered


def _heldout_categories(
    records: list[dict[str, Any]],
    *,
    ood_count: int,
    required_id_count: int,
    seed: int,
    explicit: tuple[str, ...] | None,
) -> tuple[str, ...]:
    categories = sorted({str(record["category"]) for record in records})
    if explicit is not None:
        missing = set(explicit) - set(categories)
        if missing:
            raise ValueError(f"held-out categories are absent: {sorted(missing)}")
        heldout = tuple(sorted(set(explicit)))
    else:
        ranked = sorted(
            (category for category in categories if category.lower() != "unknown"),
            key=lambda category: _rank(seed, category, "heldout_category"),
        )
        heldout_list: list[str] = []
        for category in ranked:
            remaining_id = sum(
                str(record["category"]) not in {*heldout_list, category}
                for record in records
            )
            if remaining_id < required_id_count:
                continue
            heldout_list.append(category)
            available_ood = sum(
                str(record["category"]) in heldout_list for record in records
            )
            if available_ood >= ood_count:
                break
        heldout = tuple(sorted(heldout_list))
    available_ood = sum(str(record["category"]) in heldout for record in records)
    available_id = len(records) - available_ood
    if available_ood < ood_count or available_id < required_id_count:
        raise ValueError(
            "category holdout cannot satisfy the requested ID and OOD split counts"
        )
    return heldout


def create_thingi10k_manifest(
    root: Path,
    metadata_path: Path,
    *,
    train_count: int = 1_200,
    calibration_count: int = 200,
    validation_count: int = 200,
    ood_count: int = 200,
    final_count: int = 200,
    minimum_faces: int = 500,
    heldout_categories: tuple[str, ...] | None = None,
    seed: int = 2_028,
) -> dict[str, Any]:
    """Build the deterministic 2,000-model Thingi10K stability manifest."""

    split_counts = {
        "train": train_count,
        "calibration": calibration_count,
        "validation": validation_count,
        "final": final_count,
    }
    required_id = sum(split_counts.values())
    records = discover_thingi10k_meshes(
        root, metadata_path, minimum_faces=minimum_faces
    )
    heldout = _heldout_categories(
        records,
        ood_count=ood_count,
        required_id_count=required_id,
        seed=seed,
        explicit=heldout_categories,
    )
    id_records = [record for record in records if record["category"] not in heldout]
    ood_records = [record for record in records if record["category"] in heldout]
    id_categories = sorted({str(record["category"]) for record in id_records})
    if len(id_categories) > train_count:
        raise ValueError("train_count cannot cover every in-distribution category")
    training_anchors = [
        min(
            (record for record in id_records if record["category"] == category),
            key=lambda record: _rank(seed, category, str(record["model_id"])),
        )
        for category in id_categories
    ]
    anchor_ids = {str(record["model_id"]) for record in training_anchors}
    remaining_id = [
        record for record in id_records if str(record["model_id"]) not in anchor_ids
    ]
    training_extra, remaining_id = _stratified_take(
        remaining_id,
        train_count - len(training_anchors),
        seed=seed,
        namespace="train",
        stratum_fields=("size_proxy", "genus_proxy"),
    )
    training_records = [*training_anchors, *training_extra]
    training_records.sort(
        key=lambda record: (str(record["category"]), str(record["model_id"]))
    )
    evaluation_counts = {
        split: count for split, count in split_counts.items() if split != "train"
    }
    splits = _partition_records(
        remaining_id,
        evaluation_counts,
        seed=seed,
        stratum_fields=("size_proxy", "genus_proxy"),
    )
    splits["train"] = [
        {**record, "manifest_role": "train"} for record in training_records
    ]
    selected_ood, _ = _stratified_take(
        ood_records,
        ood_count,
        seed=seed,
        namespace="ood",
        stratum_fields=("size_proxy", "genus_proxy"),
    )
    splits["ood"] = [{**record, "manifest_role": "ood"} for record in selected_ood]
    splits = {split: splits[split] for split in BENCHMARK_SPLITS}
    return {
        "version": 1,
        "dataset": "Thingi10K",
        "seed": seed,
        "source": {
            "url": THINGI10K_URL,
            "metadata_sha256": file_sha256(metadata_path),
            "redistribution": "prohibited",
        },
        "required_splits": list(BENCHMARK_SPLITS),
        "selection": {
            "minimum_non_degenerate_faces": minimum_faces,
            "stratification": ["size_proxy", "genus_proxy"],
            "size_proxy_bins": ["500:1999", "2000:7999", "8000:inf"],
            "genus_proxy_policy": (
                "closed meshes use (2*components-euler)/2; open or missing "
                "topology metadata is a separate stratum"
            ),
            "heldout_categories": list(heldout),
            "total_selected": sum(len(value) for value in splits.values()),
        },
        "sampling": {
            "source_target": "independent_area_weighted_mesh_surface_samples",
            "normalization": "mesh_bbox_center_then_unit_max_vertex_radius",
            "degenerate_faces": "drop_double_area_at_or_below_1e-12",
        },
        "splits": splits,
    }


def discover_scanobjectnn_clouds(
    root: Path,
    files: tuple[str, ...],
    *,
    official_split: str,
    category_names: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Expand local ScanObjectNN HDF5/NPZ containers into manifest records."""

    root = root.resolve()
    records: list[dict[str, Any]] = []
    for relative_name in files:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"ScanObjectNN file path must be relative: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"ScanObjectNN container is absent: {path}")
        points, labels = load_pointcloud_container(path)
        if labels is None:
            raise ValueError(f"ScanObjectNN container has no label dataset: {path}")
        digest = file_sha256(path)
        for index, label_value in enumerate(labels.tolist()):
            label = int(label_value)
            if category_names is not None and not 0 <= label < len(category_names):
                raise ValueError(
                    f"ScanObjectNN label is outside category names: {label}"
                )
            category = (
                category_names[label]
                if category_names is not None
                else f"label_{label:02d}"
            )
            records.append(
                {
                    "category": category,
                    "category_label": label,
                    "model_id": f"{official_split}:{relative.stem}:{index:06d}",
                    "pointcloud": relative.as_posix(),
                    "pointcloud_sha256": digest,
                    "record_index": index,
                    "point_count": int(points.shape[1]),
                    "official_split": official_split,
                    "normals_estimated": True,
                    "normal_method": "pca_knn",
                }
            )
    if not records:
        raise FileNotFoundError("no ScanObjectNN records were discovered")
    return records


def create_scanobjectnn_manifest(
    root: Path,
    *,
    train_files: tuple[str, ...] = ("training_objectdataset.h5",),
    test_files: tuple[str, ...] = ("test_objectdataset.h5",),
    train_count: int = 1_000,
    calibration_count: int = 150,
    validation_count: int = 100,
    ood_count: int = 200,
    final_count: int = 100,
    heldout_categories: tuple[str, ...] | None = None,
    category_names: tuple[str, ...] | None = None,
    seed: int = 2_029,
) -> dict[str, Any]:
    """Build official-train/test and category-OOD ScanObjectNN partitions."""

    training = discover_scanobjectnn_clouds(
        root,
        train_files,
        official_split="train",
        category_names=category_names,
    )
    testing = discover_scanobjectnn_clouds(
        root,
        test_files,
        official_split="test",
        category_names=category_names,
    )
    heldout = _heldout_categories(
        testing,
        ood_count=ood_count,
        required_id_count=validation_count + final_count,
        seed=seed,
        explicit=heldout_categories,
    )
    training_id = [record for record in training if record["category"] not in heldout]
    testing_id = [record for record in testing if record["category"] not in heldout]
    testing_ood = [record for record in testing if record["category"] in heldout]
    if len(training_id) < train_count + calibration_count:
        raise ValueError("held-out categories leave too few official training records")
    train_splits = _partition_records(
        training_id,
        {"train": train_count, "calibration": calibration_count},
        seed=seed,
        stratum_fields=("category",),
    )
    test_splits = _partition_records(
        testing_id,
        {"validation": validation_count, "final": final_count},
        seed=seed,
        stratum_fields=("category",),
    )
    selected_ood, _ = _stratified_take(
        testing_ood,
        ood_count,
        seed=seed,
        namespace="ood",
        stratum_fields=("category",),
    )
    splits = {
        "train": train_splits["train"],
        "calibration": train_splits["calibration"],
        "validation": test_splits["validation"],
        "ood": [{**record, "manifest_role": "ood"} for record in selected_ood],
        "final": test_splits["final"],
    }
    return {
        "version": 1,
        "dataset": "ScanObjectNN",
        "seed": seed,
        "source": {
            "url": SCANOBJECTNN_URL,
            "access": "manual_form_gated_download",
            "redistribution": "prohibited",
        },
        "required_splits": list(BENCHMARK_SPLITS),
        "selection": {
            "official_train_files": list(train_files),
            "official_test_files": list(test_files),
            "heldout_categories": list(heldout),
            "category_stratified": True,
            "total_selected": sum(len(value) for value in splits.values()),
        },
        "sampling": {
            "points_per_role": 2_048,
            "source_fresh_policy": (
                "partition finite scan indices into disjoint halves, then sample "
                "each role deterministically with replacement when needed"
            ),
            "normalization": "cloud_bbox_center_then_unit_max_point_radius",
            "normals": "estimated_pca_knn",
            "normals_estimated": True,
        },
        "splits": splits,
    }


def _categories(value: str) -> tuple[str, ...]:
    categories = tuple(part.strip() for part in value.split(",") if part.strip())
    if not categories:
        raise argparse.ArgumentTypeError("provide one or more comma-separated synsets")
    return categories


def _files(value: str) -> tuple[str, ...]:
    files = tuple(part.strip() for part in value.split(",") if part.strip())
    if not files:
        raise argparse.ArgumentTypeError("provide one or more comma-separated files")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("shapenetcore", "modelnet40", "thingi10k", "scanobjectnn"),
        default="shapenetcore",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-categories", type=_categories)
    parser.add_argument("--heldout-categories", type=_categories)
    parser.add_argument("--heldout-category-count", type=int, default=8)
    parser.add_argument("--train-per-category", type=int, default=32)
    parser.add_argument("--calibration-per-category", type=int, default=0)
    parser.add_argument("--validation-per-category", type=int, default=8)
    parser.add_argument("--category-ood-per-category", type=int, default=8)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--calibration-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--ood-count", type=int)
    parser.add_argument("--final-count", type=int)
    parser.add_argument("--minimum-faces", type=int, default=500)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--train-files", type=_files)
    parser.add_argument("--test-files", type=_files)
    parser.add_argument("--category-names", type=_categories)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    if args.dataset == "thingi10k":
        if args.metadata is None:
            parser.error("Thingi10K requires --metadata")
        manifest = create_thingi10k_manifest(
            args.root,
            args.metadata,
            train_count=1_200 if args.train_count is None else args.train_count,
            calibration_count=(
                200 if args.calibration_count is None else args.calibration_count
            ),
            validation_count=(
                200 if args.validation_count is None else args.validation_count
            ),
            ood_count=200 if args.ood_count is None else args.ood_count,
            final_count=200 if args.final_count is None else args.final_count,
            minimum_faces=args.minimum_faces,
            heldout_categories=args.heldout_categories,
            seed=2_028 if args.seed is None else args.seed,
        )
    elif args.dataset == "scanobjectnn":
        manifest = create_scanobjectnn_manifest(
            args.root,
            train_files=args.train_files or ("training_objectdataset.h5",),
            test_files=args.test_files or ("test_objectdataset.h5",),
            train_count=1_000 if args.train_count is None else args.train_count,
            calibration_count=(
                150 if args.calibration_count is None else args.calibration_count
            ),
            validation_count=(
                100 if args.validation_count is None else args.validation_count
            ),
            ood_count=200 if args.ood_count is None else args.ood_count,
            final_count=100 if args.final_count is None else args.final_count,
            heldout_categories=args.heldout_categories,
            category_names=args.category_names,
            seed=2_029 if args.seed is None else args.seed,
        )
    else:
        seed = 1_507 if args.seed is None else args.seed
        train_categories = args.train_categories
        heldout_categories = args.heldout_categories
        if args.dataset == "modelnet40" and heldout_categories is None:
            discovered = discover_modelnet40_meshes(args.root)
            train_categories, heldout_categories = split_modelnet40_categories(
                tuple(discovered["train"]),
                heldout_count=args.heldout_category_count,
                seed=seed,
            )
        elif args.dataset == "modelnet40" and train_categories is None:
            discovered = discover_modelnet40_meshes(args.root)
            train_categories = tuple(
                sorted(set(discovered["train"]) - set(heldout_categories))
            )
        if train_categories is None or heldout_categories is None:
            parser.error(
                "ShapeNetCore requires --train-categories and --heldout-categories"
            )

        common = {
            "root": args.root,
            "train_categories": train_categories,
            "heldout_categories": heldout_categories,
            "train_per_category": args.train_per_category,
            "validation_per_category": args.validation_per_category,
            "category_ood_per_category": args.category_ood_per_category,
            "seed": seed,
        }
        if args.dataset == "modelnet40":
            archive_sha256 = file_sha256(args.archive) if args.archive else None
            manifest = create_modelnet40_manifest(
                **common,
                calibration_per_category=args.calibration_per_category,
                archive_sha256=archive_sha256,
            )
        else:
            manifest = create_pilot_manifest(**common)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                **{
                    split: len(records) for split, records in manifest["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
