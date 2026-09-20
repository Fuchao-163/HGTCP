#!/usr/bin/env python3
"""Summarize completed graph-task JSON files as mean +/- sample std."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    for dataset in ("zinc", "mnist"):
        files = sorted((args.root / dataset).glob("seed_*.json"))
        if not files:
            print(f"{dataset}: no completed runs")
            continue
        records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        metric = "mae" if dataset == "zinc" else "accuracy"
        values = [record["test"][metric] for record in records]
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"{dataset}: {metric}={statistics.mean(values):.6f} +/- "
            f"{deviation:.6f} ({len(values)} seeds)"
        )


if __name__ == "__main__":
    main()
