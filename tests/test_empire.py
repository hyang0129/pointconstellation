import json

import pytest

from pointconstellation.cluster.empire import (
    SlurmAllocation,
    parse_all_jobs,
    parse_squeue_allocations,
    sync_node_registry,
    validate_launch,
    validate_logical_node,
)
from pointconstellation.cluster.jupyter import connection_settings


def test_parse_squeue_keeps_only_running_single_node_jupyter_jobs() -> None:
    output = "\n".join(
        [
            "100|jupyter_empire_8682|RUNNING|alphagpu04|1-00:00:00",
            "101|jupyter_empire_8683|PENDING|(Priority)|2-00:00:00",
            "102|training|RUNNING|alphagpu05|01:00:00",
            "103|jupyter_empire_8882|RUNNING|alphagpu06|01:00:00",
            "104|jupyter_empire_8684|RUNNING|alphagpu[07-08]|01:00:00",
        ]
    )

    assert parse_squeue_allocations(output) == [
        SlurmAllocation(
            "100",
            "jupyter_empire_8682",
            "RUNNING",
            "alphagpu04",
            8682,
            "1-00:00:00",
        )
    ]


def test_sync_node_registry_preserves_defaults(tmp_path) -> None:
    path = tmp_path / "nodes.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "python": "${HOME}/pointconstellation/.venv/bin/python",
                    "project_root": "${HOME}/pointconstellation",
                    "max_concurrent_jobs": 1,
                },
                "nodes": [{"name": "stale", "hostname": "old"}],
            }
        )
    )
    allocation = SlurmAllocation(
        "55", "jupyter_empire_8685", "RUNNING", "alphagpu10", 8685, "2:00:00"
    )

    summary = sync_node_registry(path, [allocation])
    written = json.loads(path.read_text())

    assert summary["added"] == ["alphagpu10-8685"]
    assert summary["removed"] == ["stale"]
    assert written["defaults"]["max_concurrent_jobs"] == 1
    assert written["nodes"][0]["jupyter_url"] == "http://alphagpu10:8685"


def test_guarded_launch_rejects_caps_and_port_collisions() -> None:
    collision = parse_all_jobs("10|jupyter_empire_8682|RUNNING|alphagpu01\n")
    with pytest.raises(RuntimeError, match="already allocated"):
        validate_launch(collision, 8682)

    capped = [
        {
            "job_id": str(index),
            "name": f"jupyter_empire_860{index}",
            "state": "RUNNING",
            "hostname": f"alphagpu{index}",
        }
        for index in range(6)
    ]
    with pytest.raises(RuntimeError, match="cap 6"):
        validate_launch(capped, 8690)


def test_hallulens_ports_do_not_consume_pointconstellation_capacity() -> None:
    hallulens = [
        {
            "job_id": str(index),
            "name": f"jupyter_empire_888{index}",
            "state": "RUNNING",
            "hostname": f"alphagpu{index}",
        }
        for index in range(6)
    ]

    validate_launch(hallulens, 8600)

    with pytest.raises(ValueError, match="between 8600 and 8699"):
        validate_launch([], 8882)


def test_registry_rejects_non_pc_ports_and_mismatched_logical_names() -> None:
    assert (
        validate_logical_node("alphagpu10-8601", "alphagpu10", "http://alphagpu10:8601")
        == 8601
    )

    with pytest.raises(ValueError, match="86xx"):
        validate_logical_node("alphagpu10-8882", "alphagpu10", "http://alphagpu10:8882")
    with pytest.raises(ValueError, match="logical node name"):
        validate_logical_node("alphagpu10", "alphagpu10", "http://alphagpu10:8601")


def test_connection_settings_require_url_and_read_password(monkeypatch) -> None:
    monkeypatch.delenv("POINTCONSTELLATION_JUPYTER_URL", raising=False)
    with pytest.raises(RuntimeError, match="JUPYTER_URL"):
        connection_settings()

    monkeypatch.setenv("POINTCONSTELLATION_JUPYTER_URL", "http://localhost:18882/")
    monkeypatch.setenv("POINTCONSTELLATION_JUPYTER_PASSWORD", "not-committed")
    assert connection_settings() == ("http://localhost:18882", "not-committed")
