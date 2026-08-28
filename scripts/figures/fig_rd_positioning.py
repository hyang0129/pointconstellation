#!/usr/bin/env python3
"""Render the measured D1/D2 rate-distortion positioning figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.bd_rate import (  # noqa: E402
    RD_METHODS,
    aggregate_rd_curve,
    select_curve_rows,
)
from pointconstellation.benchmark_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
)

METRICS = (
    ("official_d1_psnr_db", "D1 PSNR (dB)"),
    ("official_d2_psnr_db", "D2 PSNR (dB)"),
)
RATES = (
    ("rate_bpp", "Full stream (bpp)"),
    ("payload_bpp", "Payload only (bpp)"),
)
COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")


def _select_dataset(rows: list[dict[str, object]], dataset: str | None) -> str:
    available = sorted(
        {
            str(row["dataset"])
            for row in rows
            if row.get("metric_name") in {name for name, _ in METRICS}
        }
    )
    if dataset is not None:
        if dataset not in available:
            raise ValueError(f"dataset {dataset!r} is absent from the registry")
        return dataset
    for candidate in available:
        if candidate.lower() == "modelnet40":
            return candidate
    return available[0] if available else "unknown"


def _plot_curve(
    axis: object,
    curve: object,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    points = curve.points
    if not points:
        return
    rates = [point.rate for point in points]
    qualities = [point.quality for point in points]
    axis.plot(
        rates,
        qualities,
        color=color,
        linewidth=1.4,
        linestyle=linestyle,
        label=label,
        zorder=2,
    )
    lower = [point.ci_lower for point in points]
    upper = [point.ci_upper for point in points]
    if len(points) > 1 and all(value is not None for value in (*lower, *upper)):
        axis.fill_between(rates, lower, upper, color=color, alpha=0.14, linewidth=0)
    for is_dominated, facecolor in ((False, color), (True, "none")):
        selected = [point for point in points if point.dominated is is_dominated]
        if selected:
            axis.scatter(
                [point.rate for point in selected],
                [point.quality for point in selected],
                marker=marker,
                s=34,
                facecolors=facecolor,
                edgecolors=color,
                linewidths=1.1,
                zorder=3,
            )


def generate_figure(
    registry_path: Path,
    output_dir: Path,
    *,
    dataset: str | None = None,
    split: str = "validation",
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260826,
) -> tuple[Path, Path]:
    """Generate PDF and PNG artifacts from measured registry rows only."""

    import matplotlib.pyplot as plt

    rows = load_registry(registry_path)
    selected_dataset = _select_dataset(rows, dataset)
    rows = [
        row
        for row in rows
        if row.get("dataset") == selected_dataset
        and row.get("split") == split
        and row.get("metric_name") in {name for name, _ in METRICS}
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex="col")
    plotted_labels: set[str] = set()
    for metric_index, (metric_name, metric_label) in enumerate(METRICS):
        for rate_index, (rate_field, rate_label) in enumerate(RATES):
            axis = axes[metric_index][rate_index]
            panel_curves = 0
            for method_index, method in enumerate(RD_METHODS):
                method_rows = select_curve_rows(
                    rows,
                    method.aliases,
                    dataset=selected_dataset,
                    split=split,
                    metric_name=metric_name,
                    rate_field=rate_field,
                )
                if not method_rows:
                    continue
                curve = aggregate_rd_curve(
                    method_rows,
                    rate_field=rate_field,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=(
                        bootstrap_seed
                        + 10_000 * metric_index
                        + 1_000 * rate_index
                        + method_index
                    ),
                )
                if not curve.points:
                    continue
                _plot_curve(
                    axis,
                    curve,
                    label=method.label,
                    color=COLORS[method_index],
                    marker=MARKERS[method_index],
                )
                plotted_labels.add(method.label)
                panel_curves += 1

                if method.key == "gpcc" and rate_field == "rate_bpp":
                    amortized_rows = select_curve_rows(
                        rows,
                        method.aliases,
                        dataset=selected_dataset,
                        split=split,
                        metric_name=metric_name,
                        rate_field="amortized_stream_bpp",
                    )
                    if amortized_rows:
                        amortized = aggregate_rd_curve(
                            amortized_rows,
                            rate_field="amortized_stream_bpp",
                            bootstrap_samples=bootstrap_samples,
                            bootstrap_seed=bootstrap_seed + 50_000 + metric_index,
                        )
                        _plot_curve(
                            axis,
                            amortized,
                            label="G-PCC (SPS/GPS amortized)",
                            color=COLORS[method_index],
                            marker=MARKERS[method_index],
                            linestyle="--",
                        )
                        plotted_labels.add("G-PCC (SPS/GPS amortized)")
                        panel_curves += 1
            if not panel_curves:
                axis.text(
                    0.5,
                    0.5,
                    "No measured registry points",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_xscale("log")
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.set_title(rate_label)
            axis.set_ylabel(metric_label)
            if metric_index == len(METRICS) - 1:
                axis.set_xlabel("Bits per input point")

    handles = []
    labels = []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label in plotted_labels and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(handles)),
            frameon=False,
        )
    figure.suptitle(f"{selected_dataset} {split}: measured RD positioning")
    figure.text(
        0.5,
        0.035,
        "Hollow markers are dominated measured points; bands are 95% "
        "seed/cloud bootstrap intervals.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_rd_positioning.pdf"
    png_path = output_dir / "fig_rd_positioning.png"
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
    parser.add_argument("--split", default="validation")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    args = parser.parse_args()
    pdf_path, png_path = generate_figure(
        args.registry,
        args.output_dir,
        dataset=args.dataset,
        split=args.split,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path)}))


if __name__ == "__main__":
    main()
