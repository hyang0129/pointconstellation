#!/usr/bin/env python3
"""Plot serialized bytes against accuracy from the benchmark registry."""

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

_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


def _marker_sizes(values: list[float | None]) -> list[float]:
    finite = [value for value in values if value is not None]
    if not finite:
        return [70.0] * len(values)
    lower, upper = min(finite), max(finite)
    if upper == lower:
        return [95.0 if value is not None else 55.0 for value in values]
    return [
        45.0 + 135.0 * (value - lower) / (upper - lower) if value is not None else 55.0
        for value in values
    ]


def generate_figure(
    registry_path: Path,
    output_dir: Path,
    *,
    dataset: str | None = None,
    split: str | None = None,
) -> tuple[Path, Path]:
    """Generate PDF and PNG rate--utility plots from registry rows only."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    statistics = rate_utility_statistics(
        load_registry(registry_path), dataset=dataset, split=split
    )
    points = [
        point for point in statistics["points"] if point.get("accuracy") is not None
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    if not points:
        axis.text(0.5, 0.5, "No rate--accuracy results", ha="center", va="center")
    else:
        distortion_metric = (
            "d1_rmse"
            if any(point.get("d1_rmse") is not None for point in points)
            else "surface_rmse"
        )
        distortions = [
            float(point[distortion_metric])
            if point.get(distortion_metric) is not None
            else None
            for point in points
        ]
        sizes = _marker_sizes(distortions)
        finite_distortions = [value for value in distortions if value is not None]
        normalizer = None
        if finite_distortions:
            lower, upper = min(finite_distortions), max(finite_distortions)
            if lower == upper:
                padding = max(abs(lower) * 0.01, 1e-12)
                lower, upper = lower - padding, upper + padding
            normalizer = mpl.colors.Normalize(vmin=lower, vmax=upper)

        families = sorted({str(point["representation_family"]) for point in points})
        palette = plt.get_cmap("tab10")
        for family_index, family in enumerate(families):
            family_points = sorted(
                (
                    (point, size, distortion)
                    for point, size, distortion in zip(
                        points, sizes, distortions, strict=True
                    )
                    if point["representation_family"] == family
                ),
                key=lambda item: float(item[0]["rate_bytes"]),
            )
            x_values = [float(item[0]["rate_bytes"]) for item in family_points]
            y_values = [float(item[0]["accuracy"]) for item in family_points]
            line_color = palette(family_index % 10)
            if len(family_points) > 1:
                axis.plot(x_values, y_values, color=line_color, alpha=0.55, linewidth=1)
            available = [item for item in family_points if item[2] is not None]
            if available:
                axis.scatter(
                    [float(item[0]["rate_bytes"]) for item in available],
                    [float(item[0]["accuracy"]) for item in available],
                    c=[float(item[2]) for item in available],
                    cmap="viridis_r",
                    norm=normalizer,
                    s=[item[1] for item in available],
                    marker=_MARKERS[family_index % len(_MARKERS)],
                    edgecolors=line_color,
                    linewidths=1.1,
                    label=family,
                    zorder=3,
                )
            missing = [item for item in family_points if item[2] is None]
            if missing:
                axis.scatter(
                    [float(item[0]["rate_bytes"]) for item in missing],
                    [float(item[0]["accuracy"]) for item in missing],
                    c="#d0d0d0",
                    s=[item[1] for item in missing],
                    marker=_MARKERS[family_index % len(_MARKERS)],
                    edgecolors=line_color,
                    linewidths=1.1,
                    label=family if not available else None,
                    zorder=3,
                )
        if normalizer is not None:
            colorbar = figure.colorbar(
                mpl.cm.ScalarMappable(norm=normalizer, cmap="viridis_r"), ax=axis
            )
            distortion_label = (
                "Official D1 RMSE" if distortion_metric == "d1_rmse" else "Surface RMSE"
            )
            colorbar.set_label(f"{distortion_label} (marker colour and size)")
        axis.legend(title="Representation family", frameon=False)

    axis.set_xlabel("Serialized bytes per cloud")
    axis.set_ylabel("Accuracy")
    dataset_label = statistics["dataset"] or "No matching dataset"
    split_label = statistics["split"] or "no matching split"
    axis.set_title(f"{dataset_label}: rate--utility ({split_label})")
    axis.grid(alpha=0.25, linewidth=0.6)
    figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig_rate_utility.pdf"
    png_path = output_dir / "fig_rate_utility.png"
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
