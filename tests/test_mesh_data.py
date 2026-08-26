from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.data.mesh import (
    MeshSurfaceDataset,
    file_sha256,
    load_mesh,
    load_mesh_manifest,
    load_obj,
    load_off,
    normalize_mesh,
)
from pointconstellation.mesh_manifest import (
    build_final_slice_manifest,
    create_modelnet40_manifest,
    create_pilot_manifest,
    discover_modelnet40_meshes,
    split_modelnet40_categories,
)

FIXTURE_ROOT = Path("tests/fixtures/meshes")
FIXTURE_MANIFEST = Path("tests/fixtures/mesh_manifest.json")


def test_obj_loading_normalization_and_surface_sampling_are_deterministic() -> None:
    path = FIXTURE_ROOT / "000001/train_a/models/model_normalized.obj"
    mesh = normalize_mesh(load_obj(path))
    dataset = MeshSurfaceDataset(
        FIXTURE_ROOT,
        FIXTURE_MANIFEST,
        split="train",
        num_points=64,
        seed=19,
    )

    first = dataset.sample(0)
    repeated = dataset.sample(0)

    assert mesh.faces.shape[1] == 3
    assert np.linalg.norm(mesh.vertices, axis=1).max() == pytest.approx(1.0)
    assert np.array_equal(first.source_points, repeated.source_points)
    assert np.array_equal(first.target_points, repeated.target_points)
    assert not np.array_equal(first.source_points, first.target_points)
    assert np.linalg.norm(first.source_points, axis=1).max() <= 1.0 + 1e-6
    assert np.allclose(np.linalg.norm(first.source_normals, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(first.target_normals, axis=1), 1.0)


def test_off_loader_triangulates_polygons_and_accepts_inline_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quad.off"
    path.write_text(
        """OFF 4 1 0
# vertex colors after xyz are ignored
0 0 0 255 0 0
1 0 0 0 255 0
1 1 0 0 0 255
0 1 0 255 255 255
4 0 1 2 3
"""
    )

    mesh = load_off(path)

    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert np.array_equal(load_mesh(path).faces, mesh.faces)

    concatenated = tmp_path / "concatenated.off"
    concatenated.write_text(path.read_text().replace("OFF 4", "OFF4", 1))
    assert np.array_equal(load_off(concatenated).faces, mesh.faces)


def test_off_loader_rejects_out_of_range_indices(tmp_path: Path) -> None:
    path = tmp_path / "bad.off"
    path.write_text("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 3\n")

    with pytest.raises(ValueError, match="out of range"):
        load_off(path)


def test_training_target_defaults_to_encoder_visible_source() -> None:
    pytest.importorskip("torch")
    source_target = MeshSurfaceDataset(
        FIXTURE_ROOT,
        FIXTURE_MANIFEST,
        split="validation",
        num_points=32,
        seed=23,
        training_target="source",
    )[0]
    independent_target = MeshSurfaceDataset(
        FIXTURE_ROOT,
        FIXTURE_MANIFEST,
        split="validation",
        num_points=32,
        seed=23,
        training_target="independent",
    )[0]

    assert source_target["source_points"].equal(source_target["target_points"])
    assert source_target["fresh_points"].equal(independent_target["target_points"])
    assert not independent_target["source_points"].equal(
        independent_target["target_points"]
    )


def test_manifest_rejects_cross_split_identity(tmp_path: Path) -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text())
    manifest["splits"]["category_ood"][0] = manifest["splits"]["train"][0]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="multiple splits"):
        load_mesh_manifest(path)


def test_dataset_detects_mesh_hash_drift(tmp_path: Path) -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text())
    manifest["splits"]["train"][0]["mesh_sha256"] = "0" * 64
    path = tmp_path / "bad-hash.json"
    path.write_text(json.dumps(manifest))
    dataset = MeshSurfaceDataset(
        FIXTURE_ROOT,
        path,
        split="train",
        num_points=32,
        seed=29,
    )

    with pytest.raises(ValueError, match="hash differs"):
        dataset.sample(0)


def test_pilot_manifest_selection_is_deterministic_and_disjoint() -> None:
    options = {
        "train_categories": ("000001",),
        "heldout_categories": ("000002",),
        "train_per_category": 1,
        "validation_per_category": 1,
        "category_ood_per_category": 1,
        "seed": 31,
    }

    first = create_pilot_manifest(FIXTURE_ROOT, **options)
    second = create_pilot_manifest(FIXTURE_ROOT, **options)
    identities = [
        f"{record['category']}/{record['model_id']}"
        for records in first["splits"].values()
        for record in records
    ]

    assert first == second
    assert len(identities) == len(set(identities)) == 3


def _write_off(path: Path, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "OFF\n"
        "4 4 0\n"
        f"0 0 0\n{scale} 0 0\n0 {scale} 0\n0 0 {scale}\n"
        "3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n"
    )


def test_modelnet_manifest_preserves_official_splits_and_category_ood(
    tmp_path: Path,
) -> None:
    for category in ("alpha", "beta"):
        for official_split in ("train", "test"):
            for index in range(3):
                _write_off(
                    tmp_path
                    / category
                    / official_split
                    / f"{category}_{index:04d}.off",
                    float(index + 1),
                )

    discovered = discover_modelnet40_meshes(tmp_path)
    manifest = create_modelnet40_manifest(
        tmp_path,
        train_categories=("alpha",),
        heldout_categories=("beta",),
        train_per_category=1,
        calibration_per_category=1,
        validation_per_category=1,
        category_ood_per_category=1,
        seed=37,
        archive_sha256="a" * 64,
    )

    assert set(discovered) == {"train", "test"}
    assert manifest["dataset"] == "ModelNet40"
    assert manifest["source"]["archive_sha256"] == "a" * 64
    assert manifest["splits"]["train"][0]["official_split"] == "train"
    assert manifest["splits"]["calibration"][0]["official_split"] == "train"
    assert (
        manifest["splits"]["train"][0]["model_id"]
        != manifest["splits"]["calibration"][0]["model_id"]
    )
    assert manifest["splits"]["validation"][0]["official_split"] == "test"
    assert manifest["splits"]["category_ood"][0]["category"] == "beta"


def test_modelnet_category_split_is_deterministic_and_disjoint() -> None:
    categories = tuple(f"class_{index}" for index in range(10))

    train, heldout = split_modelnet40_categories(categories, heldout_count=3, seed=41)
    repeated = split_modelnet40_categories(categories, heldout_count=3, seed=41)

    assert (train, heldout) == repeated
    assert len(train) == 7
    assert len(heldout) == 3
    assert set(train).isdisjoint(heldout)
    assert set(train) | set(heldout) == set(categories)


def test_final_slice_manifest_excludes_prior_meshes_and_stratifies(
    tmp_path: Path,
) -> None:
    categories = ("alpha", "beta", "gamma", "delta")
    for category in categories:
        _write_off(
            tmp_path / "meshes" / category / "train" / f"{category}_train.off",
            1.0,
        )
        for index in range(3):
            _write_off(
                tmp_path / "meshes" / category / "test" / f"{category}_{index:04d}.off",
                float(index + 1),
            )

    discovered = discover_modelnet40_meshes(tmp_path / "meshes")
    category_partition = {
        "train": ["alpha", "beta"],
        "heldout": ["delta", "gamma"],
    }
    input_paths = []
    for manifest_index, selected_categories in enumerate(
        (("alpha", "gamma"), ("beta", "delta"))
    ):
        records = [discovered["test"][category][0] for category in selected_categories]
        path = tmp_path / f"input_{manifest_index}.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "dataset": "ModelNet40",
                    "categories": category_partition,
                    "splits": {"used": records},
                }
            )
        )
        input_paths.append(path)

    manifest = build_final_slice_manifest(
        tmp_path / "meshes",
        input_manifests=input_paths,
        validation_cap_per_category=1,
        minimum_ood_clouds=4,
        seed=43,
    )

    excluded = {
        (record["category"], record["model_id"])
        for record in manifest["excluded_meshes"]
    }
    selected = {
        (record["category"], record["model_id"])
        for records in manifest["splits"].values()
        for record in records
    }
    assert excluded == {
        ("alpha", "alpha_0000"),
        ("beta", "beta_0000"),
        ("gamma", "gamma_0000"),
        ("delta", "delta_0000"),
    }
    assert excluded.isdisjoint(selected)
    assert [
        record["category"] for record in manifest["splits"]["final_validation"]
    ] == ["alpha", "beta"]
    assert {
        category: sum(
            record["category"] == category for record in manifest["splits"]["final_ood"]
        )
        for category in ("delta", "gamma")
    } == {"delta": 2, "gamma": 2}
    assert manifest["input_manifests"] == [
        {"path": path.as_posix(), "sha256": file_sha256(path)} for path in input_paths
    ]
