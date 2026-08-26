from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.data.mesh import file_sha256
from pointconstellation.data.pointcloud import (
    RawPointCloudDataset,
    load_pointcloud_manifest,
)
from pointconstellation.mesh_manifest import create_scanobjectnn_manifest


def _manifest_record(path: Path, *, label: int = 3) -> dict[str, object]:
    return {
        "category": "chair",
        "category_label": label,
        "model_id": "test:tiny:000000",
        "pointcloud": path.name,
        "pointcloud_sha256": file_sha256(path),
        "record_index": 0,
        "normals_estimated": True,
    }


def test_scanobjectnn_hdf5_adapter_has_disjoint_fresh_role_and_estimated_normals(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    torch = pytest.importorskip("torch")
    path = tmp_path / "tiny.h5"
    points = np.random.default_rng(53).normal(size=(2, 64, 3)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=points)
        handle.create_dataset("label", data=np.asarray([[3], [4]], dtype=np.int64))
    manifest = {
        "version": 1,
        "dataset": "ScanObjectNN",
        "splits": {"validation": [_manifest_record(path)]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dataset = RawPointCloudDataset(
        tmp_path,
        manifest_path,
        split="validation",
        num_points=32,
        seed=59,
        normal_neighbors=8,
    )

    sample = dataset.sample(0)
    item = dataset[0]

    assert set(sample.source_indices).isdisjoint(sample.fresh_indices)
    assert sample.source_points.shape == sample.fresh_points.shape == (32, 3)
    assert np.allclose(np.linalg.norm(sample.source_normals, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(sample.fresh_normals, axis=1), 1.0)
    assert item["normals_estimated"] is True
    assert item["normal_method"] == "pca_knn"
    assert item["category_label"] == 3
    assert torch.equal(item["source_points"], item["target_points"])
    assert not torch.equal(item["source_points"], item["fresh_points"])


def test_scanobjectnn_npz_manifest_builder_preserves_roles_and_hashes(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(61)
    points = rng.normal(size=(8, 32, 3)).astype(np.float32)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    np.savez(tmp_path / "training.npz", data=points, label=labels)
    np.savez(tmp_path / "test.npz", data=points + 0.1, label=labels)

    manifest = create_scanobjectnn_manifest(
        tmp_path,
        train_files=("training.npz",),
        test_files=("test.npz",),
        train_count=2,
        calibration_count=1,
        validation_count=1,
        ood_count=2,
        final_count=1,
        heldout_categories=("label_01",),
        seed=67,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    loaded = load_pointcloud_manifest(manifest_path)
    records = [record for split in loaded["splits"].values() for record in split]

    assert tuple(loaded["splits"]) == (
        "train",
        "calibration",
        "validation",
        "ood",
        "final",
    )
    assert len(loaded["splits"]["ood"]) == 2
    assert all(record["normals_estimated"] for record in records)
    assert all(len(record["pointcloud_sha256"]) == 64 for record in records)
    assert all(record["category"] == "label_01" for record in loaded["splits"]["ood"])
    assert all(
        record["official_split"] == "train"
        for split in ("train", "calibration")
        for record in loaded["splits"][split]
    )
    dataset = RawPointCloudDataset(
        tmp_path,
        manifest_path,
        split="validation",
        num_points=16,
        seed=71,
        normal_neighbors=4,
    )
    sample = dataset.sample(0)
    assert sample.source_points.shape == sample.fresh_points.shape == (16, 3)
    assert set(sample.source_indices).isdisjoint(sample.fresh_indices)
    assert np.allclose(np.linalg.norm(sample.fresh_normals, axis=1), 1.0)
    assert np.array_equal(sample.source_points, dataset.sample(0).source_points)
