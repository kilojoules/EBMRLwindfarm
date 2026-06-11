#!/usr/bin/env python3
"""Print compact summaries for budgeted long-horizon control result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(f"\n{path}")
    print("-" * len(str(path)))
    for method_name, summary in payload.get("eval", {}).items():
        reward = summary.get("reward_mean", float("nan"))
        cost = summary.get("cost_mean", float("nan"))
        satisfied = summary.get("satisfied_mean", float("nan"))
        utilization = summary.get("utilization_mean", float("nan"))
        print(
            f"{method_name:>18}  "
            f"reward={reward:9.2f}  "
            f"cost={cost:8.2f}  "
            f"sat={satisfied:5.2f}  "
            f"util={utilization:6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for raw_path in args.paths:
        summarize(Path(raw_path))


if __name__ == "__main__":
    main()

