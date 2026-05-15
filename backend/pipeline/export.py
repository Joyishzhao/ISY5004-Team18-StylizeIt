"""Stage 7 - Export.

Encode the composited frames into the final MP4 the API will serve.
"""

from __future__ import annotations

from pathlib import Path

import cv2


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    output_dir: Path = ctx.run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final.mp4"

    frame_paths = sorted(ctx.composited_dir.glob("*.png"))
    if not frame_paths:
        # Dump folder counts so the user can see exactly which stage failed.
        run_dir = ctx.run_dir
        snapshot = {}
        for sub in run_dir.iterdir() if run_dir.exists() else []:
            if sub.is_dir():
                snapshot[sub.name] = len(list(sub.glob("*.png")))
        raise RuntimeError(
            f"[export] No composited frames in {ctx.composited_dir}. "
            f"Run-dir snapshot: {snapshot}"
        )

    sample = cv2.imread(str(frame_paths[0]))
    height, width = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, ctx.fps or 24.0, (width, height))

    try:
        for fp in frame_paths:
            writer.write(cv2.imread(str(fp)))
    finally:
        writer.release()

    ctx.output_path = output_path
