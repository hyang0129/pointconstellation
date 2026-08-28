#!/usr/bin/env python3
"""Plot official D1 distortion against accuracy per objective and regime."""

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


def _pareto_frontier(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return points not dominated by lower D1 and higher accuracy."""

    frontier = []
    best_accuracy = float("-inf")
    ordered = sorted(
        points,
        key=lambda point: (float(point["d1_rmse"]), -float(point["accuracy"])),
    )
    for point in ordered:
        accuracy = float(point["accuracy"])
        if accuracy > best_accuracy:
            frontier.append(point)
            best_accuracy = accuracy
    return frontier


def generate_figure(
    registry_path: Path,
    output_dir: Path,
    *,
    dataset: str | None = None,
    split: str | None = None,
) -> tuple[Path, Path]:
    """Generate PDF and PNG objective Pareto plots from registry rows only."""

    import matplotlib.pyplot as plt

    statistics = rate_utility_statistics(
        load_registry(registry_path), dataset=dataset, split=split
    )
    points = [
        point
        for point in statistics["points"]
        if point.get("accuracy") is not None and point.get("d1_rmse") is not None
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    if not points:
        axis.text(
            0.5,
            0.5,
            "No paired D1--accuracy results",
            ha="center",
            va="center",
        )
    else:
        group_keys = sorted(
            {
                (
                    str(point.get("objective") or "unspecified objective"),
                    str(point.get("regime") or "unspecified regime"),
                )
                for point in points
            }
        )
        palette = plt.get_cmap("tab10")
        for group_index, (objective, regime) in enumerate(group_keys):
            selected = [
                point
                for point in points
                if str(point.get("objective") or "unspecified objective") == objective
                and str(point.get("regime") or "unspecified regime") == regime
            ]
            color = palette(group_index % 10)
            axis.scatter(
                [float(point["d1_rmse"]) for point in selected],
                [float(point["accuracy"]) for point in selected],
                s=62,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                label=f"{objective} / {regime}",
                zorder=3,
            )
            frontier = _pareto_frontier(selected)
            if len(frontier) > 1:
                axis.plot(
                    [float(point["d1_rmse"]) for point in frontier],
                    [float(point["accuracy"]) for point in frontier],
                    color=color,
                    linewidth=1.2,
                    alpha=0.7,
                )
        axis.legend(title="Objective / regime", frameon=False)

    axis.set_xlabel("Official D1 RMSE")
    axis.set_ylabel("Accuracy")
    dataset_label = statistics["dataset"] or "No matching dataset"
    split_label = statistics["split"] or "no matching split"
    axis.set_title(f"{dataset_label}: objective trade-off ({split_label})")
    axis.grid(alpha=0.25, linewidth=0.6)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_objective_pareto.pdf"
    png_path = output_dir / "fig_objective_pareto.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/local/figures")
    )
    parser.add_argument("--dataset")
    parser.add_argument("--split")
    args = parser.parse_args()
    pdf_path, png_path = generate_figure(
        args.registry, args.output_dir, dataset=args.dataset, split=args.split
    )
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path)}))


if __name__ == "__main__":
    main()
