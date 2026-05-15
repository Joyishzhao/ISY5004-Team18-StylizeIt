"""Stage 6 - Compositor.

Pastes the stylized target back onto the ORIGINAL background using the
binary mask, with a feathered edge to hide the seam.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    composited_dir: Path = ctx.run_dir / "composited"
    composited_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(ctx.frames_dir.glob("*.png"))
    stylized_paths = sorted(ctx.stylized_dir.glob("*.png"))
    mask_paths = sorted(ctx.masks_dir.glob("*.png"))

    import logging
    log = logging.getLogger("stylizeit.compositor")
    log.info(
        "[compositor] read frames=%d stylized=%d masks=%d",
        len(frame_paths), len(stylized_paths), len(mask_paths),
    )

    if not (frame_paths and stylized_paths and mask_paths):
        raise RuntimeError(
            "[compositor] One of the upstream directories is empty: "
            f"frames={ctx.frames_dir} ({len(frame_paths)}), "
            f"stylized={ctx.stylized_dir} ({len(stylized_paths)}), "
            f"masks={ctx.masks_dir} ({len(mask_paths)})."
        )

    written = 0
    for fp, sp, mp in zip(frame_paths, stylized_paths, mask_paths):
        bg = cv2.imread(str(fp))
        fg = cv2.imread(str(sp))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if bg is None or fg is None or mask is None:
            log.warning("[compositor] cv2.imread failed at %s", fp.name)
            continue

        # Align shapes defensively.
        h, w = bg.shape[:2]
        if fg.shape[:2] != (h, w):
            fg = cv2.resize(fg, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        feather = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        feather = feather[..., None]
        merged = fg.astype(np.float32) * feather + bg.astype(np.float32) * (1 - feather)
        cv2.imwrite(str(composited_dir / fp.name), merged.clip(0, 255).astype(np.uint8))
        written += 1

    if written == 0:
        raise RuntimeError(f"[compositor] Wrote 0 files to {composited_dir}.")
    log.info("[compositor] wrote %d composited PNGs.", written)
    ctx.composited_dir = composited_dir
