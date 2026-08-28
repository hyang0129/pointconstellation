#!/usr/bin/env python3
"""Generate the gated BD-rate/BD-PSNR LaTeX table from the registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pointconstellation.bd_rate import (  # noqa: E402
    RD_METHODS,
    BDBootstrapEstimate,
    BDComparison,
    bootstrap_bd_comparison,
    select_curve_rows,
)
from pointconstellation.benchmark_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
)

METRICS = ("official_d1_psnr_db", "official_d2_psnr_db")
RATE_DEFINITIONS = (
    ("Full stream", "rate_bpp"),
    ("Payload only", "payload_bpp"),
)


def _select_dataset(rows: list[dict[str, Any]], dataset: str | None) -> str:
    available = sorted(
        {str(row["dataset"]) for row in rows if row.get("metric_name") in METRICS}
    )
    if dataset is not None:
        if dataset not in available:
            raise ValueError(f"dataset {dataset!r} is absent from the registry")
        return dataset
    for candidate in available:
        if candidate.lower() == "modelnet40":
            return candidate
    return available[0] if available else "unknown"


def _latex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _cell(estimate: BDBootstrapEstimate, *, unit: str) -> str:
    if estimate.value is None or estimate.ci_lower is None or estimate.ci_upper is None:
        return r"\textit{insufficient overlap}"
    suffix = r"\%" if unit == "percent" else ""
    return (
        f"{estimate.value:.2f}{suffix} "
        f"[{estimate.ci_lower:.2f}, {estimate.ci_upper:.2f}]"
    )


def render_table(statistics: dict[str, Any]) -> str:
    """Render deterministic LaTeX with explicit cells for every failed gate."""

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Bjøntegaard results for the constellation refiner relative "
        r"to each anchor. Negative BD-rate and positive BD-PSNR favor the "
        r"refiner. Brackets are paired-cloud, decoder/refiner-seed bootstrap "
        r"95\% confidence intervals. Metrics are omitted unless both measured "
        r"curves contain at least four points with a common integration range.}",
        r"\label{tab:bd-rate}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Rate basis & Anchor & D1 BD-rate & D1 BD-PSNR & D2 BD-rate & "
        r"D2 BD-PSNR \\",
        r"\midrule",
    ]
    for row in statistics["rows"]:
        comparisons = row["comparisons"]
        d1 = comparisons["official_d1_psnr_db"]
        d2 = comparisons["official_d2_psnr_db"]
        lines.append(
            f"{_latex(row['rate_label'])} & {_latex(row['anchor_label'])} & "
            f"{_cell(d1.bd_rate, unit='percent')} & "
            f"{_cell(d1.bd_psnr, unit='db')} & "
            f"{_cell(d2.bd_rate, unit='percent')} & "
            f"{_cell(d2.bd_psnr, unit='db')} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _unavailable(samples: int, reason: str) -> BDComparison:
    estimate = BDBootstrapEstimate(
        value=None,
        ci_lower=None,
        ci_upper=None,
        reason=reason,
        bootstrap_valid=0,
        bootstrap_samples=samples,
    )
    return BDComparison(estimate, estimate, 0)


def generate_table(
    registry_path: Path,
    output_path: Path,
    *,
    dataset: str | None = None,
    split: str = "validation",
    target_method: str = "refiner",
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260826,
) -> Path:
    """Write the full-stream and payload-only BD table."""

    rows = load_registry(registry_path)
    selected_dataset = _select_dataset(rows, dataset)
    rows = [
        row
        for row in rows
        if row.get("dataset") == selected_dataset
        and row.get("split") == split
        and row.get("metric_name") in METRICS
    ]
    methods = {method.key: method for method in RD_METHODS}
    if target_method not in methods:
        raise ValueError(f"unknown target method: {target_method}")
    target = methods[target_method]
    anchors: list[tuple[str, str, tuple[str, ...], str | None]] = [
        (method.key, method.label, method.aliases, None)
        for method in RD_METHODS
        if method.key != target_method
    ]
    gpcc = methods["gpcc"]
    anchors.insert(
        next(index for index, anchor in enumerate(anchors) if anchor[0] == "gpcc") + 1,
        (
            "gpcc_amortized",
            "G-PCC (SPS/GPS amortized)",
            gpcc.aliases,
            "amortized_stream_bpp",
        ),
    )

    table_rows = []
    comparison_index = 0
    curve_cache: dict[tuple[tuple[str, ...], str, str], list[dict[str, Any]]] = {}

    def curve_rows(
        aliases: tuple[str, ...], metric_name: str, rate_field: str
    ) -> list[dict[str, Any]]:
        key = (aliases, metric_name, rate_field)
        if key not in curve_cache:
            curve_cache[key] = select_curve_rows(
                rows,
                aliases,
                dataset=selected_dataset,
                split=split,
                metric_name=metric_name,
                rate_field=rate_field,
            )
        return curve_cache[key]

    for rate_label, default_rate_field in RATE_DEFINITIONS:
        for anchor_key, anchor_label, aliases, rate_override in anchors:
            if anchor_key == "gpcc_amortized" and default_rate_field != "rate_bpp":
                continue
            anchor_rate_field = rate_override or default_rate_field
            target_rate_field = default_rate_field
            comparisons = {}
            for metric_name in METRICS:
                target_rows = curve_rows(target.aliases, metric_name, target_rate_field)
                anchor_rows = curve_rows(aliases, metric_name, anchor_rate_field)
                if target_rows and anchor_rows:
                    comparisons[metric_name] = bootstrap_bd_comparison(
                        anchor_rows,
                        target_rows,
                        rate_field=default_rate_field,
                        anchor_rate_field=anchor_rate_field,
                        test_rate_field=target_rate_field,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed + comparison_index,
                    )
                else:
                    comparisons[metric_name] = _unavailable(
                        bootstrap_samples, "one or both curves have no measured points"
                    )
                comparison_index += 1
            table_rows.append(
                {
                    "rate_label": rate_label,
                    "anchor_label": anchor_label,
                    "comparisons": comparisons,
                }
            )
    statistics = {
        "dataset": selected_dataset,
        "split": split,
        "target": target.label,
        "rows": table_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_table(statistics))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local/figures/table_bd_rate.tex"),
    )
    parser.add_argument("--dataset")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--target-method", default="refiner")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    args = parser.parse_args()
    output = generate_table(
        args.registry,
        args.output,
        dataset=args.dataset,
        split=args.split,
        target_method=args.target_method,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"table": str(output)}))


if __name__ == "__main__":
    main()
