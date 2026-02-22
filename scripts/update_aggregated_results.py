#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResultEntry:
    path: Path
    payload: dict[str, Any]
    dataset_id: int | None


def _parse_dataset_id(payload: dict[str, Any]) -> int | None:
    raw_value = payload.get("dataset_id")
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str) and raw_value.isdigit():
        return int(raw_value)
    return None


def _load_existing_order(aggregated_path: Path) -> dict[int, int]:
    try:
        with open(aggregated_path, "r", encoding="utf-8") as file:
            current_data = json.load(file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(
            f"[WARN] Could not parse existing JSON: {aggregated_path}",
            file=sys.stderr,
        )
        return {}

    if not isinstance(current_data, list):
        return {}

    order: dict[int, int] = {}
    for index, item in enumerate(current_data):
        if not isinstance(item, dict):
            continue
        dataset_id = _parse_dataset_id(item)
        if dataset_id is None or dataset_id in order:
            continue
        order[dataset_id] = index
    return order


def _collect_result_entries(aggregated_path: Path) -> tuple[list[ResultEntry], int]:
    parent_dir = aggregated_path.parent
    result_files = sorted(parent_dir.glob("*/results.json"))
    entries: list[ResultEntry] = []
    skipped = 0

    for result_file in result_files:
        try:
            with open(result_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skipping unreadable JSON {result_file}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if not isinstance(payload, dict):
            print(f"[WARN] Skipping non-object JSON {result_file}", file=sys.stderr)
            skipped += 1
            continue

        entries.append(
            ResultEntry(
                path=result_file,
                payload=payload,
                dataset_id=_parse_dataset_id(payload),
            )
        )

    return entries, skipped


def _sort_entries(entries: list[ResultEntry], existing_order: dict[int, int]) -> list[ResultEntry]:
    def sort_key(entry: ResultEntry) -> tuple[float, float, str]:
        if entry.dataset_id is not None and entry.dataset_id in existing_order:
            existing_idx = float(existing_order[entry.dataset_id])
        else:
            existing_idx = math.inf
        dataset_id_rank = float(entry.dataset_id) if entry.dataset_id is not None else math.inf
        return (existing_idx, dataset_id_rank, entry.path.as_posix())

    return sorted(entries, key=sort_key)


def _find_aggregated_paths(inputs: list[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for input_path in inputs:
        path = Path(input_path)
        if path.is_file():
            candidates = [path] if path.name == "aggregated_results.json" else []
        elif path.is_dir():
            candidates = sorted(path.rglob("aggregated_results.json"))
        else:
            print(f"[WARN] Path not found, skipping: {path}", file=sys.stderr)
            continue

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(candidate)

    return discovered


def update_aggregated_file(aggregated_path: Path, *, dry_run: bool) -> tuple[int, int]:
    entries, skipped = _collect_result_entries(aggregated_path)
    existing_order = _load_existing_order(aggregated_path)
    ordered_entries = _sort_entries(entries, existing_order)
    aggregated_payload = [entry.payload for entry in ordered_entries]

    if not dry_run:
        with open(aggregated_path, "w", encoding="utf-8") as file:
            json.dump(aggregated_payload, file, indent=2)
            file.write("\n")

    return len(aggregated_payload), skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild aggregated_results.json by scanning sibling */results.json files "
            "inside each aggregated file's parent directory."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["experiments"],
        help=(
            "One or more aggregated_results.json files or directories to scan "
            "(default: experiments)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing files.",
    )
    args = parser.parse_args()

    aggregated_paths = _find_aggregated_paths(args.paths)
    if not aggregated_paths:
        print("[WARN] No aggregated_results.json files found.", file=sys.stderr)
        return 1

    updated = 0
    total_skipped = 0

    for aggregated_path in aggregated_paths:
        count, skipped = update_aggregated_file(aggregated_path, dry_run=args.dry_run)
        updated += 1
        total_skipped += skipped
        action = "WOULD UPDATE" if args.dry_run else "UPDATED"
        print(f"[{action}] {aggregated_path}: {count} entries")
        if skipped:
            print(f"[WARN] {aggregated_path}: skipped {skipped} invalid results.json file(s)", file=sys.stderr)

    print(f"[DONE] Processed {updated} aggregated_results.json file(s)")
    if total_skipped:
        print(f"[WARN] Total skipped results.json files: {total_skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
