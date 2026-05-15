"""Stage 1 - Ingest.

Decode the uploaded video to RGB frames on disk and capture metadata
(fps / resolution / duration). Enforces MVP limits from doc/API.md.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from backend.core.config import settings


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    cap = cv2.VideoCapture(str(ctx.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {ctx.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    if duration > settings.max_duration_sec + 0.5:
        cap.release()
        raise ValueError(
            f"VIDEO_LIMIT_EXCEEDED: duration {duration:.1f}s > {settings.max_duration_sec}s"
        )
    if width > settings.max_width or height > settings.max_height:
        cap.release()
        raise ValueError(
            f"VIDEO_LIMIT_EXCEEDED: {width}x{height} > "
            f"{settings.max_width}x{settings.max_height}"
        )
    if fps > settings.max_fps + 0.5:
        cap.release()
        raise ValueError(f"VIDEO_LIMIT_EXCEEDED: fps {fps:.1f} > {settings.max_fps}")

    frames_dir: Path = ctx.run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), frame)
        idx += 1
    cap.release()

    if idx == 0:
        raise RuntimeError(
            f"[ingest] No frames decoded from {ctx.video_path}. Is the file a valid video?"
        )

    import logging
    logging.getLogger("stylizeit.ingest").info(
        "[ingest] wrote %d frames at %dx%d @ %.2f fps.", idx, width, height, fps
    )

    ctx.frames_dir = frames_dir
    ctx.fps = fps
    ctx.width = width
    ctx.height = height
