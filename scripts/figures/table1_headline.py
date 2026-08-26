#!/usr/bin/env python3
"""Generate the headline LaTeX table from the benchmark registry only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.benchmark_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    HEADLINE_METHODS,
    headline_statistics,
    load_registry,
)


def _cell(statistic: dict[str, object] | None) -> str:
    if statistic is None:
        return r"\textemdash"
    return (
        f"{float(statistic['rmse']):.2f} "
        f"[{float(statistic['ci_lower']):.2f}, "
        f"{float(statistic['ci_upper']):.2f}]"
    )


def render_table(statistics: dict[str, object]) -> str:
    """Render deterministic LaTeX for the available headline methods."""

    panels = statistics["panels"]
    assert isinstance(panels, dict)
    validation = panels["validation"]["methods"]
    ood = panels["ood"]["methods"]
    assert isinstance(validation, dict) and isinstance(ood, dict)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Official D1 RMSE at 50 serialized bytes. Brackets give "
        r"paired-bootstrap 95\% confidence intervals.}",
        r"\label{tab:headline}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Validation & Category OOD \\",
        r"\midrule",
    ]
    for method, label in HEADLINE_METHODS:
        if method not in validation and method not in ood:
            continue
        latex_label = label.replace("-", r"\mbox{-}")
        lines.append(
            f"{latex_label} & {_cell(validation.get(method))} & "
            f"{_cell(ood.get(method))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def generate_table(
    registry_path: Path,
    output_path: Path,
    *,
    dataset: str | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260824,
) -> Path:
    rows = load_registry(registry_path)
    statistics = headline_statistics(
        rows,
        dataset=dataset,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
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
        default=Path("artifacts/local/figures/table1_headline.tex"),
    )
    parser.add_argument("--dataset")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    args = parser.parse_args()
    output = generate_table(
        args.registry,
        args.output,
        dataset=args.dataset,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"table": str(output)}))


if __name__ == "__main__":
    main()
