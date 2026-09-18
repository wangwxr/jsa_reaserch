#!/usr/bin/env python3
"""Launch only preregistered 8.7 Stage-1 CRAM groups after gradient audits exist."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import common


PRIMARY = ("C1", "C2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run C1/C2 and audit-authorized C3")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def audit(dataset: str) -> dict:
    path = common.HERE / "gradient_audit" / f"{dataset}_seed12345.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run the training-side gradient audit first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def groups() -> tuple[str, ...]:
    audits = [audit(dataset) for dataset in common.DATASETS]
    return PRIMARY + (("C3",) if all(item["c3_allowed"] for item in audits) else ())


def complete(dataset: str, group: str) -> bool:
    return common.checkpoint_path(dataset, group, 12345, "best").is_file() and common.checkpoint_path(dataset, group, 12345, "final").is_file()


def main() -> None:
    args = parse_args()
    selected = groups()
    if args.status:
        print(json.dumps({"groups": selected, "stage2_allowed": False, "completed": {f"{dataset}/{group}": complete(dataset, group) for dataset in common.DATASETS for group in selected}}, indent=2))
        return
    if not args.run:
        raise RuntimeError("Choose --status or --run")
    # Fixed waves protect shared GPU memory; C0 is a read-only checkpoint reuse.
    waves = (("vggss", "C1", 0), ("vggss", "C2", 1), ("flickr", "C1", 0), ("flickr", "C2", 1))
    if "C3" in selected:
        waves += (("vggss", "C3", 0), ("flickr", "C3", 1))
    for dataset, group, gpu in waves:
        if complete(dataset, group):
            print(f"completed: {dataset}/{group}", flush=True); continue
        subprocess.run([sys.executable, "train.py", "--dataset", dataset, "--group", group, "--seed", "12345", "--gpu", str(gpu)], check=True)
    print("8.7 Stage-1 training matrix complete; Stage 2 remains forbidden pending evaluation gate.", flush=True)


if __name__ == "__main__":
    main()
