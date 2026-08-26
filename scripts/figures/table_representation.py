#!/usr/bin/env python3
"""Generate the 50-byte representation-metrics table from the registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.benchmark_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
    rate_utility_statistics,
)


def _latex(value: object | None) -> str:
    if value is None:
        return r"\textemdash"
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _metric(point: dict[str, object], name: str, *, digits: int = 4) -> str:
    value = point.get(name)
    return r"\textemdash" if value is None else f"{float(value):.{digits}f}"


def render_table(statistics: dict[str, object]) -> str:
    """Render deterministic LaTeX for every available 50-byte arm."""

    points = statistics["points"]
    assert isinstance(points, list)
    dataset = _latex(statistics.get("dataset") or "unspecified dataset")
    split = _latex(statistics.get("split") or "unspecified split")
    rate_bytes = float(statistics.get("rate_bytes") or 50.0)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            f"\\caption{{Representation metrics at {rate_bytes:g} serialized bytes on "
            f"{dataset} ({split}). Values are means over available registry rows; "
            r"D1 RMSE is aggregated in squared-error space.}"
        ),
        r"\label{tab:representation}",
        r"\begin{tabular}{llllrrrrrr}",
        r"\toprule",
        (
            r"Representation & Arm & Objective & Regime & D1 RMSE & Accuracy & mAP "
            r"& Repeat. & Surface RMSE & Normal cons. \\"
        ),
        r"\midrule",
    ]
    for point in points:
        arm = str(point["method"])
        if point.get("arm_label") not in (None, "default"):
            arm = f"{arm} / {point['arm_label']}"
        lines.append(
            " & ".join(
                (
                    _latex(point.get("representation_family")),
                    _latex(arm),
                    _latex(point.get("objective")),
                    _latex(point.get("regime")),
                    _metric(point, "d1_rmse", digits=3),
                    _metric(point, "accuracy"),
                    _metric(point, "map"),
                    _metric(point, "repeatability"),
                    _metric(point, "surface_rmse"),
                    _metric(point, "normal_consistency"),
                )
            )
            + r" \\"
        )
    if not points:
        lines.append("\\multicolumn{10}{c}{No matching 50-byte rows} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def generate_table(
    registry_path: Path,
    output_path: Path,
    *,
    dataset: str | None = None,
    split: str | None = None,
    rate_bytes: float = 50.0,
) -> Path:
    statistics = rate_utility_statistics(
        load_registry(registry_path),
        dataset=dataset,
        split=split,
        rate_bytes=rate_bytes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_table(statistics))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local/figures/table_representation.tex"),
    )
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    parser.add_argument("--rate-bytes", type=float, default=50.0)
    args = parser.parse_args()
    output = generate_table(
        args.registry,
        args.output,
        dataset=args.dataset,
        split=args.split,
        rate_bytes=args.rate_bytes,
    )
    print(json.dumps({"table": str(output)}))


if __name__ == "__main__":
    main()
