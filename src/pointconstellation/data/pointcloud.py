"""Manifest-backed raw point clouds with disjoint source and fresh roles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pointconstellation.data.mesh import file_sha256

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class RawPointCloudSample:
    source_points: FloatArray
    source_normals: FloatArray
    fresh_points: FloatArray
    fresh_normals: FloatArray
    source_indices: IntArray
    fresh_indices: IntArray
    category: str
    category_label: int
    model_id: str
    sample_id: int
    normals_estimated: bool = True


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "HDF5 point clouds require h5py; install pointconstellation[datasets]"
        ) from exc
    return h5py


def _point_array(value: Any, path: Path) -> FloatArray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim == 2:
        points = points[None, ...]
    if points.ndim != 3 or points.shape[2] < 3:
        raise ValueError(f"point-cloud data must have shape (B, N, 3+): {path}")
    points = points[:, :, :3]
    if not np.isfinite(points).all():
        raise ValueError(f"point-cloud data contains non-finite coordinates: {path}")
    return points


def load_pointcloud_container(
    path: Path, *, points_key: str = "data", label_key: str = "label"
) -> tuple[FloatArray, IntArray | None]:
    """Read a ScanObjectNN-style HDF5 or NPZ point-cloud container."""

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        h5py = _h5py()
        with h5py.File(path, "r") as handle:
            if points_key not in handle:
                raise ValueError(f"point dataset {points_key!r} is absent: {path}")
            points = _point_array(handle[points_key][...], path)
            labels = handle[label_key][...] if label_key in handle else None
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            available_points_key = points_key
            if available_points_key not in archive and points_key == "data":
                available_points_key = "points"
            if available_points_key not in archive:
                raise ValueError(f"point dataset {points_key!r} is absent: {path}")
            points = _point_array(archive[available_points_key], path)
            available_label_key = label_key
            if available_label_key not in archive and label_key == "label":
                available_label_key = "labels"
            labels = archive.get(available_label_key)
    else:
        raise ValueError(f"unsupported point-cloud format {suffix!r}: {path}")

    if labels is None:
        return points, None
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(label_array) != len(points):
        raise ValueError(f"point and label record counts differ: {path}")
    return points, label_array


def normalize_point_cloud(points: FloatArray) -> FloatArray:
    """Apply the mesh benchmark's bounding-box center and unit-radius scale."""

    if points.ndim != 2 or points.shape != (len(points), 3):
        raise ValueError("point cloud must have shape (N, 3)")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    centered = points - 0.5 * (lower + upper)
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-12:
        raise ValueError("point cloud has zero spatial extent")
    return (centered / scale).astype(np.float32)


def estimate_pca_knn_normals(
    points: FloatArray, *, neighbors: int = 16, chunk_size: int = 256
) -> FloatArray:
    """Estimate deterministic unoriented normals from local PCA neighborhoods."""

    if points.ndim != 2 or points.shape != (len(points), 3):
        raise ValueError("point cloud must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains non-finite coordinates")
    if not 3 <= neighbors < len(points):
        raise ValueError("normal neighbors must be in [3, number of points)")
    if chunk_size < 1:
        raise ValueError("normal estimation chunk size must be positive")

    points64 = points.astype(np.float64, copy=False)
    squared_norms = np.einsum("ij,ij->i", points64, points64)
    normals = np.empty_like(points64)
    for start in range(0, len(points64), chunk_size):
        stop = min(start + chunk_size, len(points64))
        query = points64[start:stop]
        distances = (
            np.einsum("ij,ij->i", query, query)[:, None]
            + squared_norms[None, :]
            - 2.0 * query @ points64.T
        )
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
        neighborhoods = points64[indices]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors
        _, eigenvectors = np.linalg.eigh(covariance)
        normals[start:stop] = eigenvectors[:, :, 0]

    # PCA normals have an arbitrary sign. Canonicalizing the largest-magnitude
    # component makes manifests and tests reproducible; D2 itself is sign-free.
    axes = np.abs(normals).argmax(axis=1)
    signs = normals[np.arange(len(normals)), axes] < 0
    normals[signs] *= -1.0
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= 1e-12):
        raise ValueError("PCA normal estimation produced an invalid normal")
    return (normals / lengths[:, None]).astype(np.float32)


def load_pointcloud_manifest(path: Path) -> dict[str, Any]:
    """Validate a version-1 raw-point-cloud manifest."""

    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("point-cloud manifest version must be 1")
    if not isinstance(manifest.get("dataset"), str):
        raise ValueError("point-cloud manifest must declare a dataset")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("point-cloud manifest must contain nonempty splits")
    required_splits = manifest.get("required_splits", [])
    if not isinstance(required_splits, list) or any(
        not isinstance(split, str) for split in required_splits
    ):
        raise ValueError("point-cloud manifest required_splits must be a list")
    missing_splits = set(required_splits) - splits.keys()
    if missing_splits:
        raise ValueError(f"point-cloud manifest is missing splits: {missing_splits}")

    seen: set[str] = set()
    for split, records in splits.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"manifest split must be a nonempty list: {split}")
        for record in records:
            required = {
                "category",
                "category_label",
                "model_id",
                "pointcloud",
                "pointcloud_sha256",
                "record_index",
                "normals_estimated",
            }
            if not isinstance(record, dict) or not required <= record.keys():
                raise ValueError(f"invalid record in manifest split: {split}")
            if record["normals_estimated"] is not True:
                raise ValueError("raw point-cloud normals must be marked estimated")
            identity = str(record["model_id"])
            if identity in seen:
                raise ValueError(
                    f"point-cloud identity appears in multiple splits: {identity}"
                )
            seen.add(identity)
    return manifest


def _role_seed(seed: int, split: str, model_id: str, role: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{model_id}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class RawPointCloudDataset:
    """PyTorch-compatible raw clouds with disjoint finite-scan resampling roles."""

    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        *,
        split: str,
        num_points: int = 2048,
        seed: int,
        normal_neighbors: int = 16,
        verify_hashes: bool = True,
    ) -> None:
        if num_points < 8:
            raise ValueError("num_points must be at least 8")
        if normal_neighbors < 3:
            raise ValueError("normal_neighbors must be at least 3")
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_pointcloud_manifest(self.manifest_path)
        if split not in self.manifest["splits"]:
            raise ValueError(f"split is absent from point-cloud manifest: {split}")
        self.split = split
        self.records = self.manifest["splits"][split]
        self.num_points = num_points
        self.seed = seed
        self.normal_neighbors = normal_neighbors
        self.verify_hashes = verify_hashes
        self._container_cache: dict[Path, tuple[FloatArray, IntArray | None]] = {}
        self._geometry_cache: dict[tuple[Path, int], tuple[FloatArray, FloatArray]] = {}
        self._verified_paths: set[Path] = set()

    def __len__(self) -> int:
        return len(self.records)

    def _pointcloud_path(self, record: dict[str, Any]) -> Path:
        relative = Path(record["pointcloud"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manifest point-cloud path must be relative: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"manifest point-cloud escapes dataset root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest point-cloud does not exist: {path}")
        return path

    def _geometry(self, record: dict[str, Any]) -> tuple[FloatArray, FloatArray]:
        path = self._pointcloud_path(record)
        if path not in self._verified_paths:
            if self.verify_hashes and file_sha256(path) != record["pointcloud_sha256"]:
                raise ValueError(f"point-cloud hash differs from manifest: {path}")
            self._verified_paths.add(path)
        if path not in self._container_cache:
            self._container_cache[path] = load_pointcloud_container(
                path,
                points_key=str(record.get("points_key", "data")),
                label_key=str(record.get("label_key", "label")),
            )
        clouds, labels = self._container_cache[path]
        record_index = int(record["record_index"])
        if not 0 <= record_index < len(clouds):
            raise ValueError(f"point-cloud record index is out of range: {path}")
        expected_label = int(record["category_label"])
        if labels is not None and int(labels[record_index]) != expected_label:
            raise ValueError(f"point-cloud label differs from manifest: {path}")
        key = (path, record_index)
        if key not in self._geometry_cache:
            points = normalize_point_cloud(clouds[record_index])
            normals = estimate_pca_knn_normals(points, neighbors=self.normal_neighbors)
            self._geometry_cache[key] = points, normals
        return self._geometry_cache[key]

    def sample(self, index: int) -> RawPointCloudSample:
        record = self.records[index]
        points, normals = self._geometry(record)
        if len(points) < 2 * self.normal_neighbors:
            raise ValueError(
                "point cloud must contain at least twice normal_neighbors points "
                "for disjoint source/fresh roles"
            )
        model_id = str(record["model_id"])
        partition_rng = np.random.default_rng(
            _role_seed(self.seed, self.split, model_id, "partition")
        )
        permutation = partition_rng.permutation(len(points))
        midpoint = len(points) // 2
        source_pool = permutation[:midpoint]
        fresh_pool = permutation[midpoint:]

        def draw(pool: IntArray, role: str) -> IntArray:
            rng = np.random.default_rng(
                _role_seed(self.seed, self.split, model_id, role)
            )
            return np.asarray(
                rng.choice(
                    pool,
                    size=self.num_points,
                    replace=len(pool) < self.num_points,
                ),
                dtype=np.int64,
            )

        source_indices = draw(source_pool, "source")
        fresh_indices = draw(fresh_pool, "fresh")
        return RawPointCloudSample(
            source_points=points[source_indices].copy(),
            source_normals=normals[source_indices].copy(),
            fresh_points=points[fresh_indices].copy(),
            fresh_normals=normals[fresh_indices].copy(),
            source_indices=source_indices,
            fresh_indices=fresh_indices,
            category=str(record["category"]),
            category_label=int(record["category_label"]),
            model_id=model_id,
            sample_id=index,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        sample = self.sample(index)
        # The category label is metadata for downstream Track B tasks. Training
        # code consumes source_points alone and never forwards this label, the
        # normals, or the finite-scan indices to an encoder or decoder.
        return {
            "source_points": torch.from_numpy(sample.source_points.copy()),
            "source_normals": torch.from_numpy(sample.source_normals.copy()),
            "target_points": torch.from_numpy(sample.source_points.copy()),
            "target_normals": torch.from_numpy(sample.source_normals.copy()),
            "fresh_points": torch.from_numpy(sample.fresh_points.copy()),
            "fresh_normals": torch.from_numpy(sample.fresh_normals.copy()),
            "category": sample.category,
            "family": sample.category,
            "category_label": sample.category_label,
            "label": sample.category_label,
            "model_id": sample.model_id,
            "sample_id": sample.sample_id,
            "normals_estimated": sample.normals_estimated,
            "normal_method": "pca_knn",
            "fresh_disjoint": True,
        }
