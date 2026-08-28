from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from pointconstellation import final_slice
from pointconstellation.data import file_sha256
from pointconstellation.final_slice import (
    FinalSliceAlreadyCompletedError,
    FinalSliceConfig,
    run_final_slice,
)


def _record(category: str, model_id: str) -> dict[str, str]:
    return {
        "category": category,
        "model_id": model_id,
        "mesh": f"{category}/test/{model_id}.off",
        "mesh_sha256": "a" * 64,
        "official_split": "test",
    }


def test_final_slice_second_invocation_fails_without_touching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "final.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": "ModelNet40",
                "splits": {
                    "final_validation": [_record("alpha", "alpha_0001")],
                    "final_ood": [_record("beta", "beta_0001")],
                },
            }
        )
    )
    official_path = tmp_path / "official.json"
    official_path.write_text("{}\n")
    config = FinalSliceConfig(
        final_slice_manifest=str(manifest_path),
        official_stability_config=str(official_path),
        output_root=str(tmp_path / "outputs"),
    )
    calls = []

    def fake_run(official: object, *, device_name: str | None = None) -> dict:
        calls.append((official, device_name))
        output_dir = Path(official.output_dir)  # type: ignore[attr-defined]
        (output_dir / "official_metrics.json").write_text('{"completed": true}\n')
        return {"completed": True}

    monkeypatch.setattr(final_slice, "run_official_stability", fake_run)
    monkeypatch.setattr(final_slice, "_git_commit", lambda: "b" * 40)
    monkeypatch.setattr(final_slice, "_timestamp", lambda: "2026-08-26T17:00:00+00:00")

    assert run_final_slice(config) == {"completed": True}
    manifest_hash = file_sha256(manifest_path)
    output_dir = Path(config.output_root) / manifest_hash
    lock_path = output_dir / "FINAL_SLICE_LOCK"
    assert json.loads(lock_path.read_text()) == {
        "manifest_sha256": manifest_hash,
        "git_commit": "b" * 40,
        "timestamp": "2026-08-26T17:00:00+00:00",
    }
    snapshots = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output_dir.iterdir()
        if path.is_file()
    }

    with pytest.raises(FinalSliceAlreadyCompletedError, match="already completed"):
        run_final_slice(config)

    assert len(calls) == 1
    assert snapshots == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in snapshots
    }


def test_final_slice_cli_rejects_cloud_limit() -> None:
    with pytest.raises(SystemExit) as error:
        final_slice.main(
            [
                "--config",
                "unused.json",
                "--max-clouds-per-split",
                "2",
            ]
        )

    assert error.value.code == 2
