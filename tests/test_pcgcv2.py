from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pointconstellation.codecs.external import run_external_codec
from pointconstellation.codecs.pcgcv2 import (
    PCGCV2_PAYLOAD_SUFFIXES,
    PCGCV2_UPSTREAM_COMMIT,
    Pcgcv2HarnessConfig,
    Pcgcv2RatePoint,
    pack_pcgcv2_payloads,
    pcgcv2_diversity_summary,
    point_set_sha256,
    unpack_pcgcv2_payloads,
)
from pointconstellation.pcgcv2_benchmark import evaluate_pcgcv2_rate_point
from pointconstellation.pcgcv2_training import Pcgcv2RetrainConfig, _quantize


def _harness(tmp_path: Path, adapter: Path) -> Pcgcv2HarnessConfig:
    workspace = tmp_path / "workspace"
    upstream = workspace / "upstream"
    upstream.mkdir(parents=True)
    checkpoint = upstream / "ckpts/r3_0.10bpp.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"deployment weights")
    environment = workspace / "environment.json"
    environment.write_text(json.dumps({"fixture": True}))
    python = workspace / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    return Pcgcv2HarnessConfig(
        workspace_root=str(workspace),
        upstream_subpath="upstream",
        environment_python_subpath="env/bin/python",
        environment_manifest_subpath="environment.json",
        adapter_script=str(adapter),
        position_bits=(6, 7, 8),
        rate_points=(
            Pcgcv2RatePoint(
                "released_low",
                "upstream/ckpts/r3_0.10bpp.pth",
                0.5,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            ),
        ),
    )


def _fake_adapter(path: Path) -> Path:
    path.write_text(
        """from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('mode')
parser.add_argument('--upstream-dir')
parser.add_argument('--checkpoint')
parser.add_argument('--work-dir')
parser.add_argument('--position-bits')
parser.add_argument('--scaling-factor')
parser.add_argument('--rho')
parser.add_argument('--input')
parser.add_argument('--stream')
parser.add_argument('--reconstruction')
args = parser.parse_args()
if args.mode == 'encode':
    Path(args.stream).write_bytes(Path(args.input).read_bytes())
else:
    Path(args.reconstruction).write_bytes(Path(args.stream).read_bytes())
"""
    )
    return path


def _fake_pc_error(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "print('mseF      (p2point): 1.25')\n"
        "print('mseF,PSNR (p2point): 42.5')\n"
        "print('mseF      (p2plane): 0.75')\n"
        "print('mseF,PSNR (p2plane): 45.0')\n"
        "print('h.        (p2point): 3.0')\n"
        "print('h.,PSNR   (p2point): 35.0')\n"
        "print('h.        (p2plane): 2.0')\n"
        "print('h.,PSNR   (p2plane): 38.0')\n"
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_pcgcv2_framing_counts_all_components_and_round_trips(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "encoded"
    expected = []
    for index, suffix in enumerate(PCGCV2_PAYLOAD_SUFFIXES, start=1):
        payload = bytes([index]) * (index + 2)
        Path(str(prefix) + suffix).write_bytes(payload)
        expected.append(payload)

    stream = tmp_path / "stream.bin"
    payload_bytes = pack_pcgcv2_payloads(prefix, stream)
    decoded_prefix = tmp_path / "decoded"
    outputs = unpack_pcgcv2_payloads(stream, decoded_prefix)

    assert payload_bytes == stream.stat().st_size
    assert payload_bytes == 8 + 8 * 4 + sum(map(len, expected))
    assert [path.read_bytes() for path in outputs] == expected


def test_pcgcv2_framing_rejects_unaccounted_bytes(tmp_path: Path) -> None:
    prefix = tmp_path / "encoded"
    for suffix in PCGCV2_PAYLOAD_SUFFIXES:
        Path(str(prefix) + suffix).write_bytes(b"x")
    stream = tmp_path / "stream.bin"
    pack_pcgcv2_payloads(prefix, stream)
    stream.write_bytes(stream.read_bytes() + b"trailing")

    with pytest.raises(ValueError, match="trailing"):
        unpack_pcgcv2_payloads(stream, tmp_path / "decoded")


def test_pcgcv2_external_spec_runs_fake_codec_and_counts_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _fake_adapter(tmp_path / "fake_pcgcv2.py")
    harness = _harness(tmp_path, adapter)
    monkeypatch.setattr(
        "pointconstellation.codecs.external._commit",
        lambda _: PCGCV2_UPSTREAM_COMMIT,
    )
    monkeypatch.setattr(
        "pointconstellation.codecs.external._checkout_diff_sha256",
        lambda _: hashlib.sha256(b"").hexdigest(),
    )
    spec = harness.external_spec("released_low", 6)
    points = np.asarray([[-1.0, 0.0, 1.0], [0.25, -0.5, 0.75]], dtype=np.float32)

    result = run_external_codec(spec, points, work_dir=tmp_path / "run")

    assert result.stream_bytes == (tmp_path / "run/stream.bin").stat().st_size
    assert spec.model_bytes == len(b"deployment weights")
    assert result.environment == {"fixture": True}
    assert np.allclose(result.reconstruction, points, atol=1 / 63)
    assert result.decompress_command[2] == "decode"


def test_pcgcv2_benchmark_row_has_registry_rate_and_model_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _fake_adapter(tmp_path / "fake_pcgcv2.py")
    metric = _fake_pc_error(tmp_path / "pc_error")
    harness = _harness(tmp_path, adapter)
    monkeypatch.setattr(
        "pointconstellation.codecs.external._commit",
        lambda _: PCGCV2_UPSTREAM_COMMIT,
    )
    monkeypatch.setattr(
        "pointconstellation.codecs.external._checkout_diff_sha256",
        lambda _: hashlib.sha256(b"").hexdigest(),
    )
    source = np.asarray([[-1.0, 0.0, 1.0], [0.25, -0.5, 0.75]], dtype=np.float32)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1))

    row = evaluate_pcgcv2_rate_point(
        harness,
        "released_low",
        6,
        source,
        normals,
        pc_error_executable=metric,
        work_dir=tmp_path / "benchmark",
        split="ood",
        family="table",
        model_id="table_0001",
        sample_id=4,
    )

    assert row["family"] == "table"
    assert row["codec_family"] == "pcgcv2"
    assert row["payload_bytes"] == row["stream_bytes"] > 0
    assert row["model_bytes"] == len(b"deployment weights")
    assert row["model_sha256"] == hashlib.sha256(b"deployment weights").hexdigest()


def test_pcgcv2_diversity_check_rejects_constant_output() -> None:
    collapsed = [
        {
            "stream_sha256": f"{index:064x}",
            "reconstruction_sha256": "f" * 64,
            "decoded_points": 10,
        }
        for index in range(10)
    ]
    diverse = [
        {
            "stream_sha256": f"{index:064x}",
            "reconstruction_sha256": f"{index + 20:064x}",
            "decoded_points": 10,
        }
        for index in range(10)
    ]

    assert pcgcv2_diversity_summary(collapsed)["rate_point_valid"] is False
    assert pcgcv2_diversity_summary(collapsed)["constant_output"] is True
    assert pcgcv2_diversity_summary(diverse)["rate_point_valid"] is True


def test_pcgcv2_geometry_hash_ignores_point_order() -> None:
    points = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)

    assert point_set_sha256(points) == point_set_sha256(points[::-1])


def test_checked_pcgcv2_configs_pin_low_rate_and_exact_split() -> None:
    harness = Pcgcv2HarnessConfig.from_json(
        Path("configs/external/pcgcv2_low_rate.json")
    )
    retrain = Pcgcv2RetrainConfig.from_json(
        Path("configs/experiment_027_pcgcv2_retrain.json")
    )

    assert harness.position_bits == (6, 7, 8)
    assert harness.upstream_commit == PCGCV2_UPSTREAM_COMMIT
    assert len(harness.rate_points) == 3
    assert {arm.position_bits for arm in retrain.arms} == {6, 7, 8}
    assert retrain.expected_stability_manifest_sha256 == (
        "d44014dd2313b8815562cde9df2ba1927e1110fcfbc218428a7db39ef6b829ac"
    )


def test_pcgcv2_training_quantization_is_coordinate_only() -> None:
    points = np.asarray(
        [[-1.0, -0.5, 0.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
        dtype=np.float32,
    )

    quantized = _quantize(points, 6)

    assert quantized.shape == (2, 3)
    assert quantized.min() == 0
    assert quantized.max() == 63


def test_pcgcv2_training_controller_does_not_import_torch() -> None:
    source_root = Path(__file__).parents[1] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import pointconstellation.pcgcv2_training; "
        "assert 'torch' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
