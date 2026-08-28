"""Deterministic independent surface samples from manifest-listed meshes."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TriangleMesh:
    vertices: FloatArray
    faces: IntArray


@dataclass(frozen=True)
class MeshNormalization:
    """Isotropic transform between a mesh's source and normalized frames."""

    center: FloatArray
    scale: float

    def normalize(self, points: NDArray[np.floating[Any]]) -> FloatArray:
        return ((points - self.center) / self.scale).astype(np.float32)

    def restore(self, points: NDArray[np.floating[Any]]) -> FloatArray:
        return (points * self.scale + self.center).astype(np.float32)


@dataclass(frozen=True)
class MeshSurfaceSample:
    source_points: FloatArray
    source_normals: FloatArray
    target_points: FloatArray
    target_normals: FloatArray
    category: str
    model_id: str
    sample_id: int
    normalization_center: FloatArray
    normalization_scale: float
    original_source_points: FloatArray
    original_target_points: FloatArray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_obj(path: Path) -> TriangleMesh:
    """Load vertices and polygon faces from the geometry subset of OBJ."""

    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    lines = path.read_text(errors="strict").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        fields = raw_line.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f" and len(fields) >= 4:
            if not vertices:
                raise ValueError(f"face precedes vertices at {path}:{line_number}")
            indices = []
            for field in fields[1:]:
                token = field.split("/", 1)[0]
                if not token:
                    raise ValueError(f"missing face index at {path}:{line_number}")
                index = int(token)
                index = index - 1 if index > 0 else len(vertices) + index
                if index < 0 or index >= len(vertices):
                    raise ValueError(f"face index out of range at {path}:{line_number}")
                indices.append(index)
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    if len(vertices) < 3 or not faces:
        raise ValueError(f"OBJ must contain vertices and faces: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float32)
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"OBJ contains non-finite vertices: {path}")
    return TriangleMesh(vertex_array, np.asarray(faces, dtype=np.int64))


def load_off(path: Path) -> TriangleMesh:
    """Load polygon geometry from an Object File Format (OFF) mesh."""

    lines = []
    for raw_line in path.read_text(errors="strict").splitlines():
        content = raw_line.split("#", 1)[0].strip()
        if content:
            lines.append(content)
    if not lines:
        raise ValueError(f"OFF file is empty: {path}")

    header = lines.pop(0).split()
    if header[0] == "OFF":
        inline_counts = header[1:]
    elif header[0].startswith("OFF") and header[0][3:].isdigit():
        # A small number of files in the official ModelNet40 archive omit the
        # newline/space between the OFF magic and vertex count.
        inline_counts = [header[0][3:], *header[1:]]
    else:
        raise ValueError(f"OFF header is missing: {path}")
    if not inline_counts:
        if not lines:
            raise ValueError(f"OFF counts are missing: {path}")
        counts = lines.pop(0).split()
    else:
        counts = inline_counts
    if len(counts) < 2:
        raise ValueError(f"OFF counts are invalid: {path}")
    vertex_count, face_count = int(counts[0]), int(counts[1])
    if vertex_count < 3 or face_count < 1:
        raise ValueError(f"OFF must contain vertices and faces: {path}")
    if len(lines) < vertex_count + face_count:
        raise ValueError(f"OFF geometry is truncated: {path}")

    vertices: list[list[float]] = []
    for index, line in enumerate(lines[:vertex_count], 1):
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"OFF vertex is invalid at {path}:{index}")
        vertices.append([float(value) for value in fields[:3]])

    faces: list[tuple[int, int, int]] = []
    for offset, line in enumerate(lines[vertex_count : vertex_count + face_count], 1):
        fields = line.split()
        if not fields:
            raise ValueError(f"OFF face is invalid at {path}:{offset}")
        polygon_size = int(fields[0])
        if polygon_size < 3 or len(fields) < polygon_size + 1:
            raise ValueError(f"OFF face is invalid at {path}:{offset}")
        indices = [int(value) for value in fields[1 : polygon_size + 1]]
        if min(indices) < 0 or max(indices) >= vertex_count:
            raise ValueError(f"OFF face index is out of range at {path}:{offset}")
        for triangle_offset in range(1, polygon_size - 1):
            faces.append(
                (indices[0], indices[triangle_offset], indices[triangle_offset + 1])
            )

    vertex_array = np.asarray(vertices, dtype=np.float32)
    if not np.isfinite(vertex_array).all():
        raise ValueError(f"OFF contains non-finite vertices: {path}")
    return TriangleMesh(vertex_array, np.asarray(faces, dtype=np.int64))


def filter_degenerate_faces(
    mesh: TriangleMesh, *, area_epsilon: float = 1e-12
) -> TriangleMesh:
    """Drop triangles whose doubled area is at most ``area_epsilon``.

    Vertices are deliberately not compacted. Keeping the original vertex array
    makes the operation auditable and avoids changing valid face indices.
    """

    if area_epsilon < 0:
        raise ValueError("area_epsilon cannot be negative")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1:] != (3,):
        raise ValueError("mesh vertices must have shape (N, 3)")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1:] != (3,):
        raise ValueError("mesh faces must have shape (F, 3)")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("mesh contains non-finite vertices")
    if len(mesh.faces) == 0:
        return TriangleMesh(mesh.vertices.copy(), mesh.faces.copy())
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ValueError("mesh face index is out of range")
    triangles = mesh.vertices[mesh.faces]
    double_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    return TriangleMesh(
        mesh.vertices.copy(), mesh.faces[double_areas > area_epsilon].copy()
    )


def _stl_mesh(triangles: NDArray[np.float32], path: Path) -> TriangleMesh:
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"STL triangles have an invalid shape: {path}")
    if len(triangles) == 0:
        raise ValueError(f"STL contains no triangles: {path}")
    if not np.isfinite(triangles).all():
        raise ValueError(f"STL contains non-finite vertices: {path}")
    vertices = triangles.reshape(-1, 3).astype(np.float32, copy=True)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = filter_degenerate_faces(TriangleMesh(vertices, faces))
    if len(mesh.faces) == 0:
        raise ValueError(f"STL contains no non-degenerate triangles: {path}")
    return mesh


def _load_binary_stl(data: bytes, path: Path) -> TriangleMesh:
    if len(data) < 84:
        raise ValueError(f"binary STL is truncated: {path}")
    face_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * face_count
    if len(data) != expected_size:
        raise ValueError(f"binary STL size does not match its face count: {path}")
    facet_dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    facets = np.frombuffer(data, dtype=facet_dtype, count=face_count, offset=84)
    return _stl_mesh(np.asarray(facets["vertices"], dtype=np.float32), path)


def _load_ascii_stl(data: bytes, path: Path) -> TriangleMesh:
    try:
        lines = data.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"STL is neither valid binary nor ASCII: {path}") from exc
    triangles: list[list[list[float]]] = []
    facet_vertices: list[list[float]] | None = None
    saw_solid = False
    for line_number, raw_line in enumerate(lines, 1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        keyword = fields[0].lower()
        if keyword == "solid":
            saw_solid = True
        elif keyword == "facet":
            if facet_vertices is not None:
                raise ValueError(f"nested STL facet at {path}:{line_number}")
            facet_vertices = []
        elif keyword == "vertex":
            if facet_vertices is None or len(fields) != 4:
                raise ValueError(f"invalid STL vertex at {path}:{line_number}")
            try:
                facet_vertices.append([float(value) for value in fields[1:]])
            except ValueError as exc:
                raise ValueError(f"invalid STL vertex at {path}:{line_number}") from exc
        elif keyword == "endfacet":
            if facet_vertices is None or len(facet_vertices) != 3:
                raise ValueError(f"invalid STL facet at {path}:{line_number}")
            triangles.append(facet_vertices)
            facet_vertices = None
    if not saw_solid or not triangles or facet_vertices is not None:
        raise ValueError(f"ASCII STL has incomplete triangle geometry: {path}")
    return _stl_mesh(np.asarray(triangles, dtype=np.float32), path)


def load_stl(path: Path) -> TriangleMesh:
    """Load binary or ASCII STL and remove zero-area triangles."""

    data = path.read_bytes()
    if len(data) >= 84:
        face_count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * face_count:
            return _load_binary_stl(data, path)
    return _load_ascii_stl(data, path)


def load_mesh(path: Path) -> TriangleMesh:
    """Load one supported polygon mesh based on its filename suffix."""

    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix == ".off":
        return load_off(path)
    if suffix == ".stl":
        return load_stl(path)
    raise ValueError(f"unsupported mesh format {suffix!r}: {path}")


def mesh_normalization(mesh: TriangleMesh) -> MeshNormalization:
    """Return the benchmark's bounding-box-center, unit-radius transform."""

    lower = mesh.vertices.min(axis=0)
    upper = mesh.vertices.max(axis=0)
    center = (0.5 * (lower + upper)).astype(np.float32)
    centered = mesh.vertices - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 1e-12:
        raise ValueError("mesh has zero spatial extent")
    return MeshNormalization(center, scale)


def normalize_mesh(mesh: TriangleMesh) -> TriangleMesh:
    """Map a mesh into the benchmark's declared unit-radius coordinate domain."""

    normalization = mesh_normalization(mesh)
    return TriangleMesh(normalization.normalize(mesh.vertices), mesh.faces.copy())


def sample_mesh_surface(
    mesh: TriangleMesh, count: int, *, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Sample a triangle mesh uniformly by area with piecewise face normals."""

    if count < 1:
        raise ValueError("sample count must be positive")
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_areas = np.linalg.norm(cross, axis=1)
    valid = double_areas > 1e-12
    if not valid.any():
        raise ValueError("mesh has no non-degenerate triangles")
    triangles = triangles[valid]
    cross = cross[valid]
    double_areas = double_areas[valid]
    probabilities = double_areas / double_areas.sum()
    selected = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
    chosen = triangles[selected]
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    weights = np.column_stack((1.0 - first, first * (1.0 - second), first * second))
    points = np.einsum("ni,nij->nj", weights, chosen)
    face_normals = cross / double_areas[:, None]
    return points.astype(np.float32), face_normals[selected].astype(np.float32)


def _role_seed(seed: int, split: str, model_id: str, role: str) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{model_id}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_mesh_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("mesh manifest version must be 1")
    if not isinstance(manifest.get("dataset"), str):
        raise ValueError("mesh manifest must declare a dataset")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("mesh manifest must contain nonempty splits")
    required_splits = manifest.get("required_splits", [])
    if not isinstance(required_splits, list) or any(
        not isinstance(split, str) for split in required_splits
    ):
        raise ValueError("mesh manifest required_splits must be a list")
    missing_splits = set(required_splits) - splits.keys()
    if missing_splits:
        raise ValueError(f"mesh manifest is missing splits: {missing_splits}")
    seen: set[str] = set()
    for split, records in splits.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"manifest split must be a nonempty list: {split}")
        for record in records:
            required = {"category", "model_id", "mesh", "mesh_sha256"}
            if not isinstance(record, dict) or not required <= record.keys():
                raise ValueError(f"invalid record in manifest split: {split}")
            if manifest["dataset"] == "Thingi10K" and not isinstance(
                record.get("license"), str
            ):
                raise ValueError("Thingi10K records must declare a license")
            identity = f"{record['category']}/{record['model_id']}"
            if identity in seen:
                raise ValueError(
                    f"mesh identity appears in multiple splits: {identity}"
                )
            seen.add(identity)
    return manifest


class MeshSurfaceDataset:
    """PyTorch-compatible independent encoder/target sampling from meshes."""

    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        *,
        split: str,
        num_points: int,
        seed: int,
        verify_hashes: bool = True,
        training_target: str = "source",
    ) -> None:
        if num_points < 8:
            raise ValueError("num_points must be at least 8")
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_mesh_manifest(self.manifest_path)
        if split not in self.manifest["splits"]:
            raise ValueError(f"split is absent from mesh manifest: {split}")
        self.split = split
        self.records = self.manifest["splits"][split]
        self.num_points = num_points
        self.seed = seed
        self.verify_hashes = verify_hashes
        if training_target not in {"source", "independent"}:
            raise ValueError("training_target must be source or independent")
        self.training_target = training_target
        self._mesh_cache: dict[Path, tuple[TriangleMesh, MeshNormalization]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _mesh_path(self, record: dict[str, str]) -> Path:
        relative = Path(record["mesh"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manifest mesh path must be relative: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"manifest mesh escapes dataset root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest mesh does not exist: {path}")
        return path

    def _load_mesh(
        self, record: dict[str, str]
    ) -> tuple[TriangleMesh, MeshNormalization]:
        path = self._mesh_path(record)
        if path not in self._mesh_cache:
            if self.verify_hashes and file_sha256(path) != record["mesh_sha256"]:
                raise ValueError(f"mesh hash differs from manifest: {path}")
            source_mesh = load_mesh(path)
            normalization = mesh_normalization(source_mesh)
            self._mesh_cache[path] = (
                TriangleMesh(
                    normalization.normalize(source_mesh.vertices),
                    source_mesh.faces.copy(),
                ),
                normalization,
            )
        return self._mesh_cache[path]

    def sample(self, index: int) -> MeshSurfaceSample:
        record = self.records[index]
        mesh, normalization = self._load_mesh(record)
        source, source_normals = sample_mesh_surface(
            mesh,
            self.num_points,
            rng=np.random.default_rng(
                _role_seed(self.seed, self.split, record["model_id"], "source")
            ),
        )
        target, normals = sample_mesh_surface(
            mesh,
            self.num_points,
            rng=np.random.default_rng(
                _role_seed(self.seed, self.split, record["model_id"], "target")
            ),
        )
        return MeshSurfaceSample(
            source,
            source_normals,
            target,
            normals,
            str(record["category"]),
            str(record["model_id"]),
            index,
            normalization.center.copy(),
            normalization.scale,
            normalization.restore(source),
            normalization.restore(target),
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        sample = self.sample(index)
        target_points = (
            sample.source_points
            if self.training_target == "source"
            else sample.target_points
        )
        target_normals = (
            sample.source_normals
            if self.training_target == "source"
            else sample.target_normals
        )
        return {
            "source_points": torch.from_numpy(sample.source_points.copy()),
            "source_normals": torch.from_numpy(sample.source_normals.copy()),
            "target_points": torch.from_numpy(target_points.copy()),
            "target_normals": torch.from_numpy(target_normals.copy()),
            "fresh_points": torch.from_numpy(sample.target_points.copy()),
            "fresh_normals": torch.from_numpy(sample.target_normals.copy()),
            "normalization_center": torch.from_numpy(
                sample.normalization_center.copy()
            ),
            "normalization_scale": sample.normalization_scale,
            "original_source_points": torch.from_numpy(
                sample.original_source_points.copy()
            ),
            "original_target_points": torch.from_numpy(
                sample.original_target_points.copy()
            ),
            "category": sample.category,
            "family": sample.category,
            "model_id": sample.model_id,
            "sample_id": sample.sample_id,
        }
