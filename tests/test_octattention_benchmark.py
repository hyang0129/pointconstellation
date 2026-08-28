from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from pointconstellation.codecs import run_external_codec_batch
from pointconstellation.data import file_sha256
from pointconstellation.octattention_benchmark import (
    OctAttentionArm,
    OctAttentionConfig,
    _array_sha256,
    _depth_grid,
    octattention_codec_spec,
    octattention_diversity_contract,
)


def _fake_upstream(tmp_path: Path) -> tuple[Path, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "pointconstellation_adapter.py").write_text(
        """import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('encode', 'decode'))
parser.add_argument('--checkpoint')
parser.add_argument('--position-bits')
parser.add_argument('--depth', type=int)
parser.add_argument('--inputs', nargs='+')
parser.add_argument('--streams', nargs='+', required=True)
parser.add_argument('--reconstructions', nargs='+')
args = parser.parse_args()
if args.mode == 'encode':
    for input_path, stream_path in zip(args.inputs, args.streams):
        payload = Path(input_path).read_bytes()
        Path(stream_path).write_bytes(b'D' + bytes([args.depth]) + b'X' + payload)
else:
    for stream_path, reconstruction_path in zip(
        args.streams, args.reconstructions
    ):
        Path(reconstruction_path).write_bytes(Path(stream_path).read_bytes()[3:])
"""
    )
    subprocess.run(("git", "init", "-q", str(upstream)), check=True)
    subprocess.run(
        ("git", "-C", str(upstream), "add", "pointconstellation_adapter.py"),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.name=OctAttention Test",
            "-c",
            "user.email=octattention@example.invalid",
            "commit",
            "-qm",
            "fake codec",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(upstream), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return upstream, commit


def _config(tmp_path: Path, commit: str) -> OctAttentionConfig:
    checkpoint = tmp_path / "released.pth"
    checkpoint.write_bytes(b"shared model")
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"python": sys.version.split()[0]}))
    patch = tmp_path / "adapter.patch"
    patch.write_bytes(b"fixture patch")
    return OctAttentionConfig(
        name="octattention",
        upstream_url="https://example.invalid/OctAttention.git",
        upstream_branch="obj",
        upstream_commit=commit,
        license="Apache-2.0",
        paper="OctAttention",
        paper_url="https://example.invalid/paper",
        workspace_root=str(tmp_path),
        upstream_subpath="upstream",
        environment_python_subpath=sys.executable,
        environment_manifest_subpath="environment.json",
        patch_path=str(patch),
        patch_sha256=file_sha256(patch),
        checkout_diff_sha256=hashlib.sha256(b"").hexdigest(),
        stability_config="stability.json",
        expected_stability_manifest_sha256="1" * 64,
        training_source_subpath="training/sources",
        training_manifest_subpath="training/training_manifest.json",
        expected_training_meshes=512,
        source_points=2048,
        position_bits=12,
        depths=(4, 5, 6, 7),
        arms=(
            OctAttentionArm(
                name="pretrained_transfer",
                label="pretrained_transfer_mpeg_8i_mvub",
                checkpoint_subpath="released.pth",
                seeds=(800093,),
                training_content="fixture released training",
            ),
            OctAttentionArm(
                name="experiment_019_retrained",
                label="retrained_exact_experiment_019_train",
                checkpoint_subpath="checkpoints/seed_{seed}/encoder_final.pth",
                seeds=(7, 17, 29),
                training_content="fixture exact training",
            ),
        ),
        retrain_max_steps=100000,
        retrain_batch_size=32,
        retrain_bptt=1024,
        retrain_learning_rate=1e-3,
        pc_error_executable="pc_error",
        splits=("validation", "ood"),
        rate_corridor_bytes=(20, 200),
        hypothesis="fixture hypothesis",
        primary_metric="fixture D1",
        decision_rule="fixture decision",
        protocol_note="fixture protocol",
        timeout_seconds=30,
        output_dir="output",
    )


def test_checked_octattention_config_pins_protocol_and_patch() -> None:
    config = OctAttentionConfig.from_json(
        Path("configs/external/octattention_lowrate.json")
    )

    assert config.upstream_commit == "adb628b29abc4b160f55fe27dd43b0db7b730cac"
    assert config.depths == (4, 5, 6, 7)
    assert config.position_bits == 12
    assert config.source_points == 2048
    assert config.expected_training_meshes == 512
    assert config.patch_sha256 == file_sha256(Path(config.patch_path))
    assert {arm.label for arm in config.arms} == {
        "pretrained_transfer_mpeg_8i_mvub",
        "retrained_exact_experiment_019_train",
    }


def test_recorded_octattention_patch_has_declared_checkout_diff(
    tmp_path: Path,
) -> None:
    config = OctAttentionConfig.from_json(
        Path("configs/external/octattention_lowrate.json")
    )
    checkout = tmp_path / "checkout"
    subprocess.run(("git", "init", "-q", str(checkout)), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Patch Test",
            "-c",
            "user.email=patch@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(checkout), "apply", str(Path(config.patch_path).resolve())),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(checkout), "add", "-N", "pointconstellation_adapter.py"),
        check=True,
    )
    diff = subprocess.run(
        ("git", "-C", str(checkout), "diff", "--binary", "HEAD"),
        check=True,
        capture_output=True,
    ).stdout

    assert hashlib.sha256(diff).hexdigest() == config.checkout_diff_sha256


def test_octattention_fake_codec_counts_bytes_hashes_and_plumbs_depth(
    tmp_path: Path,
) -> None:
    _, commit = _fake_upstream(tmp_path)
    config = _config(tmp_path, commit)
    spec = octattention_codec_spec(
        config,
        arm=config.arm("pretrained_transfer"),
        seed=800093,
        depth=5,
        workspace=tmp_path,
    )
    clouds = (
        np.asarray([[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0]], dtype=np.float32),
        np.asarray([[-0.75, 0.125, 0.5], [0.75, -0.25, -1.0]], dtype=np.float32),
    )
    work_dirs = (tmp_path / "run_0", tmp_path / "run_1")

    results = run_external_codec_batch(spec, clouds, work_dirs=work_dirs)

    depth_index = results[0].compress_command.index("--depth") + 1
    assert results[0].compress_command[depth_index] == "5"
    for result, work_dir, cloud in zip(results, work_dirs, clouds, strict=True):
        assert result.stream_bytes == work_dir.joinpath("input.ply").stat().st_size + 3
        assert len(result.stream_sha256) == 64
        assert len(result.reconstruction_sha256) == 64
        assert result.upstream_commit == commit
        assert np.allclose(result.reconstruction, cloud, atol=1.0 / 4095)

    rows = [
        {
            "arm": "pretrained_transfer",
            "seed": 800093,
            "depth": 5,
            "codec_input_sha256": _array_sha256(
                _depth_grid(cloud, 5, position_bits=12)
            ),
            "stream_sha256": result.stream_sha256,
            "reconstruction_sha256": result.reconstruction_sha256,
        }
        for cloud, result in zip(clouds, results, strict=True)
    ]
    assert octattention_diversity_contract(rows)["passed"]


def test_octattention_diversity_contract_rejects_constant_codec() -> None:
    rows = [
        {
            "arm": "experiment_019_retrained",
            "seed": 7,
            "depth": 4,
            "codec_input_sha256": input_hash,
            "stream_sha256": "a" * 64,
            "reconstruction_sha256": "b" * 64,
        }
        for input_hash in ("1" * 64, "2" * 64)
    ]

    check = octattention_diversity_contract(rows)

    assert not check["passed"]
    assert check["groups"][0]["unique_codec_inputs"] == 2
    assert check["groups"][0]["unique_streams"] == 1
    assert check["groups"][0]["unique_reconstructions"] == 1


def test_octattention_rejects_undeclared_depth(tmp_path: Path) -> None:
    _, commit = _fake_upstream(tmp_path)
    config = _config(tmp_path, commit)

    with pytest.raises(ValueError, match="depth 3 is not declared"):
        octattention_codec_spec(
            config,
            arm=config.arm("pretrained_transfer"),
            seed=800093,
            depth=3,
            workspace=tmp_path,
        )
