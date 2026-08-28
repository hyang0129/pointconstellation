"""Create immutable mesh manifests for external surface benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pointconstellation.data.mesh import file_sha256, load_mesh_manifest

MODELNET40_URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"
FINAL_SLICE_INPUT_MANIFESTS = (
    Path("configs/manifests/modelnet40_pilot.local.json"),
    Path("configs/manifests/modelnet40_scale.local.json"),
    Path("configs/manifests/modelnet40_stability.local.json"),
)


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


def _modelnet40_manifest_categories(
    manifest: dict[str, Any], path: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if manifest.get("dataset") != "ModelNet40":
        raise ValueError(f"input manifest is not ModelNet40: {path}")
    categories = manifest.get("categories")
    if not isinstance(categories, dict):
        raise ValueError(f"input manifest has no category partition: {path}")
    train = tuple(categories.get("train", ()))
    heldout = tuple(categories.get("heldout", ()))
    if not train or not heldout or set(train) & set(heldout):
        raise ValueError(f"input manifest has an invalid category partition: {path}")
    if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError(f"input manifest categories must be unique: {path}")
    return tuple(sorted(train)), tuple(sorted(heldout))


def build_final_slice_manifest(
    root: Path,
    *,
    input_manifests: Sequence[Path],
    validation_cap_per_category: int,
    minimum_ood_clouds: int = 200,
    seed: int = 1517,
) -> dict[str, Any]:
    """Build an untouched ModelNet40 official-test evaluation manifest.

    Every official-test mesh named by any input manifest is excluded.  The
    remaining training-category meshes are deterministically capped within
    each category, while all remaining held-out-category meshes are retained.
    """

    if not input_manifests:
        raise ValueError("at least one input manifest is required")
    if validation_cap_per_category < 1:
        raise ValueError("validation_cap_per_category must be positive")
    if minimum_ood_clouds < 1:
        raise ValueError("minimum_ood_clouds must be positive")

    discovered = discover_modelnet40_meshes(root)
    test_by_mesh = {
        record["mesh"]: record
        for records in discovered["test"].values()
        for record in records
    }
    manifest_records: list[dict[str, str]] = []
    manifest_identities: list[dict[str, str]] = []
    expected_categories: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    seen_paths: set[Path] = set()
    excluded_paths: set[str] = set()
    for raw_path in input_manifests:
        path = Path(raw_path)
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f"input manifest is repeated: {path}")
        seen_paths.add(resolved)
        manifest = load_mesh_manifest(path)
        categories = _modelnet40_manifest_categories(manifest, path)
        if expected_categories is None:
            expected_categories = categories
        elif categories != expected_categories:
            raise ValueError("input manifest category partitions do not match")
        manifest_identities.append(
            {"path": path.as_posix(), "sha256": file_sha256(path)}
        )
        for records in manifest["splits"].values():
            for record in records:
                mesh = str(record["mesh"])
                discovered_record = test_by_mesh.get(mesh)
                if discovered_record is None:
                    continue
                if record["mesh_sha256"] != discovered_record["mesh_sha256"]:
                    raise ValueError(f"input manifest mesh hash differs: {mesh}")
                excluded_paths.add(mesh)

    assert expected_categories is not None
    train_categories, heldout_categories = expected_categories
    available_categories = set(discovered["test"])
    requested_categories = set(train_categories) | set(heldout_categories)
    missing = requested_categories - available_categories
    if missing:
        raise ValueError(f"input manifest categories are absent: {sorted(missing)}")

    final_validation: list[dict[str, str]] = []
    final_ood: list[dict[str, str]] = []
    for category in train_categories:
        available = [
            record
            for record in discovered["test"][category]
            if record["mesh"] not in excluded_paths
        ]
        ranked = sorted(
            available,
            key=lambda record: _rank(seed, category, record["model_id"]),
        )
        if not ranked:
            raise ValueError(f"no untouched official-test meshes for {category}")
        final_validation.extend(ranked[:validation_cap_per_category])
    for category in heldout_categories:
        available = [
            record
            for record in discovered["test"][category]
            if record["mesh"] not in excluded_paths
        ]
        if not available:
            raise ValueError(f"no untouched official-test meshes for {category}")
        final_ood.extend(available)
    if len(final_ood) < minimum_ood_clouds:
        raise ValueError(
            f"final_ood has {len(final_ood)} clouds; need at least {minimum_ood_clouds}"
        )

    for records in (final_validation, final_ood):
        records.sort(key=lambda record: (record["category"], record["model_id"]))
    for mesh in sorted(excluded_paths):
        manifest_records.append(test_by_mesh[mesh])
    manifest_records.sort(key=lambda record: (record["category"], record["model_id"]))

    source_values = []
    for raw_path in input_manifests:
        source = load_mesh_manifest(Path(raw_path)).get("source")
        if isinstance(source, dict):
            source_values.append(source)
    source = source_values[0] if source_values else {"url": MODELNET40_URL}
    if any(value != source for value in source_values[1:]):
        raise ValueError("input manifest source identities do not match")
    return {
        "version": 1,
        "dataset": "ModelNet40",
        "seed": seed,
        "source": source,
        "sampling": {
            "source_target": "independent_area_weighted_mesh_surface_samples",
            "normalization": "mesh_bbox_center_then_unit_max_vertex_radius",
            "official_split_policy": (
                "final_validation and final_ood use only official test meshes "
                "absent from every recorded input manifest"
            ),
        },
        "selection": {
            "validation_cap_per_category": validation_cap_per_category,
            "minimum_ood_clouds": minimum_ood_clouds,
            "all_remaining_heldout_meshes": True,
        },
        "categories": {
            "train": list(train_categories),
            "heldout": list(heldout_categories),
        },
        "input_manifests": manifest_identities,
        "excluded_meshes": manifest_records,
        "splits": {
            "final_validation": final_validation,
            "final_ood": final_ood,
        },
    }


def _categories(value: str) -> tuple[str, ...]:
    categories = tuple(part.strip() for part in value.split(",") if part.strip())
    if not categories:
        raise argparse.ArgumentTypeError("provide one or more comma-separated synsets")
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("shapenetcore", "modelnet40", "modelnet40-final-slice"),
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
    parser.add_argument("--seed", type=int, default=1507)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--input-manifest",
        action="append",
        type=Path,
        dest="input_manifests",
    )
    parser.add_argument("--validation-cap-per-category", type=int, default=16)
    parser.add_argument("--minimum-ood-clouds", type=int, default=200)
    args = parser.parse_args()

    if args.dataset == "modelnet40-final-slice":
        manifest = build_final_slice_manifest(
            args.root,
            input_manifests=(
                tuple(args.input_manifests)
                if args.input_manifests is not None
                else FINAL_SLICE_INPUT_MANIFESTS
            ),
            validation_cap_per_category=args.validation_cap_per_category,
            minimum_ood_clouds=args.minimum_ood_clouds,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        return

    train_categories = args.train_categories
    heldout_categories = args.heldout_categories
    if args.dataset == "modelnet40" and heldout_categories is None:
        discovered = discover_modelnet40_meshes(args.root)
        train_categories, heldout_categories = split_modelnet40_categories(
            tuple(discovered["train"]),
            heldout_count=args.heldout_category_count,
            seed=args.seed,
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
        "seed": args.seed,
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
