#!/usr/bin/env python3
"""Render Figure 1 from the benchmark registry only."""

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


def _plot_panel(axis: object, panel: dict[str, object], *, title: str) -> None:
    methods = panel["methods"]
    assert isinstance(methods, dict)
    ordered = [
        (method, label, methods[method])
        for method, label in HEADLINE_METHODS
        if method in methods
    ]
    if not ordered:
        axis.text(0.5, 0.5, "No 50 B official D1 results", ha="center", va="center")
        axis.set_xticks([])
        axis.set_title(title)
        return
    labels = [label for _, label, _ in ordered]
    values = [float(statistic["rmse"]) for _, _, statistic in ordered]
    lower = [
        max(value - float(statistic["ci_lower"]), 0.0)
        for value, (_, _, statistic) in zip(values, ordered, strict=True)
    ]
    upper = [
        max(float(statistic["ci_upper"]) - value, 0.0)
        for value, (_, _, statistic) in zip(values, ordered, strict=True)
    ]
    positions = list(range(len(ordered)))
    axis.bar(
        positions,
        values,
        yerr=[lower, upper],
        capsize=3,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)


def generate_figure(
    registry_path: Path,
    output_dir: Path,
    *,
    dataset: str | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260824,
) -> tuple[Path, Path]:
    """Generate the PDF and PNG Figure 1 artifacts."""

    import matplotlib.pyplot as plt

    rows = load_registry(registry_path)
    statistics = headline_statistics(
        rows,
        dataset=dataset,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    _plot_panel(axes[0], statistics["panels"]["validation"], title="Validation")
    _plot_panel(axes[1], statistics["panels"]["ood"], title="Category OOD")
    axes[0].set_ylabel("Official D1 RMSE (grid units)")
    dataset_label = statistics["dataset"] or "No matching dataset"
    figure.suptitle(f"{dataset_label}: selection baselines at 50 B")
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig1_selection_baselines.pdf"
    png_path = output_dir / "fig1_selection_baselines.png"
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
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260824)
    args = parser.parse_args()
    pdf_path, png_path = generate_figure(
        args.registry,
        args.output_dir,
        dataset=args.dataset,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path)}))


if __name__ == "__main__":
    main()
