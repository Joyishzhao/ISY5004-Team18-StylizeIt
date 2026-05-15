"""Quickly inspect a run directory and pinpoint where the pipeline died.

Usage:
    python scripts/inspect_run.py run_e1dc4d3c46f7ba3e
    python scripts/inspect_run.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_DIRS = ["frames", "frames_jpeg", "masks", "stylized", "stable", "composited", "output"]


def inspect(run_dir: Path) -> None:
    print(f"== {run_dir.name}")
    if not run_dir.exists():
        print(f"  (does not exist: {run_dir})")
        return

    counts = {}
    for sub in EXPECTED_DIRS:
        d = run_dir / sub
        if d.exists():
            n = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg"))) + len(list(d.glob("*.mp4")))
            counts[sub] = n
        else:
            counts[sub] = "-"
    width = max(len(k) for k in counts)
    for k, v in counts.items():
        flag = ""
        if isinstance(v, int):
            if v == 0:
                flag = "  <-- EMPTY (probably the broken stage)"
        print(f"  {k.ljust(width)} : {v}{flag}")

    # show one mask + frame shape so the user can compare
    try:
        import cv2  # type: ignore

        first_frame = next((run_dir / "frames").glob("*.png"), None)
        first_mask = next((run_dir / "masks").glob("*.png"), None)
        if first_frame:
            print(f"  frame shape:  {cv2.imread(str(first_frame)).shape}")
        if first_mask:
            print(f"  mask  shape:  {cv2.imread(str(first_mask), cv2.IMREAD_GRAYSCALE).shape}")
    except Exception:
        pass

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print("  metrics.json:")
        for k, v in json.loads(metrics_path.read_text(encoding="utf-8")).items():
            print(f"    {k} = {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", help="run_id (omit + use --all to inspect everything)")
    ap.add_argument("--all", action="store_true", help="inspect every run_dir under artifacts/runs/")
    ap.add_argument(
        "--artifacts-dir",
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "runs"),
        help="root of run directories",
    )
    args = ap.parse_args()

    base = Path(args.artifacts_dir)
    if args.all:
        for d in sorted(base.iterdir()) if base.exists() else []:
            if d.is_dir():
                inspect(d)
                print()
    elif args.run_id:
        inspect(base / args.run_id)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
