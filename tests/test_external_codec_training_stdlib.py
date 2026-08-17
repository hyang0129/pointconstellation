from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pointconstellation.external_codec_training import _checkpoint_step


def test_training_controller_imports_with_only_the_standard_library() -> None:
    source_root = Path(__file__).parents[1] / "src"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "import pointconstellation.external_codec_training"
    )

    completed = subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_checkpoint_step_reads_tensorflow_state(tmp_path: Path) -> None:
    assert _checkpoint_step(tmp_path) is None

    (tmp_path / "checkpoint").write_text('model_checkpoint_path: "model.ckpt-20000"\n')

    assert _checkpoint_step(tmp_path) == 20000
