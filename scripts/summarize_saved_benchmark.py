"""Summarize paired full-training-size results without rerunning experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize(path: Path) -> str:
    records = json.loads(path.read_text(encoding="utf-8"))
    comparisons = {"rmse": [], "r_auc_mse": []}
    seen = set()
    for record in records:
        dataset_id = record["dataset_id"]
        if dataset_id in seen:
            raise ValueError(f"Duplicate dataset ID: {dataset_id}")
        seen.add(dataset_id)
        result = record.get("sizes", {}).get("100", {})
        if result.get("status") != "ok":
            continue
        hmtl = result.get("hmtl", {})
        catboost = result.get("baselines", {}).get("catboost", result.get("catboost", {}))
        for metric, pairs in comparisons.items():
            if metric not in hmtl or metric not in catboost:
                continue
            left, right = float(hmtl[metric]), float(catboost[metric])
            if math.isfinite(left) and math.isfinite(right):
                pairs.append((left, right))

    lines = [
        "Full training size; lower is better. Counts describe this saved run only.",
        "| Metric | Paired datasets | HMTL lower | CatBoost lower | Ties |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, pairs in comparisons.items():
        lines.append(
            f"| {metric} | {len(pairs)} | {sum(a < b for a, b in pairs)} | "
            f"{sum(b < a for a, b in pairs)} | {sum(a == b for a, b in pairs)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1] / "experiments" /
        "automl_5sizes_v4_4workers_v2" / "aggregated_results.json",
    )
    args = parser.parse_args()
    print(summarize(args.path))


if __name__ == "__main__":
    main()
