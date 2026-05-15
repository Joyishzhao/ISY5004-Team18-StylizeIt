"""Stage 3 - Tracking + Segmentation with SAM 2.

Seeds SAM 2 with the bbox from the grounding stage and propagates a
binary mask through every frame of the clip. The result is one PNG per
frame in <run_dir>/masks/.

Setup expected (filled in by scripts/download_models.ps1):
    models/sam2/sam2.1_hiera_large.pt

Reference:
    Ravi et al. "SAM 2: Segment Anything in Images and Videos."
    Meta AI, 2024. https://github.com/facebookresearch/sam2

Fallback: a simple rectangle mask matching the initial bbox is written
for every frame so the rest of the pipeline still produces an output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from backend.core.config import settings

log = logging.getLogger("stylizeit.tracking")


def _candidate_checkpoints():
    base = settings.models_dir / "sam2"
    # ordered by preference (large -> tiny)
    return [
        (base / "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        (base / "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
        (base / "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        (base / "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ]


def _load_sam2_video_predictor():
    try:
        import torch  # type: ignore
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore

        for ckpt, cfg in _candidate_checkpoints():
            if ckpt.exists():
                device = (
                    settings.device
                    if (settings.device == "cuda" and torch.cuda.is_available())
                    or settings.device in ("mps", "cpu")
                    else "cpu"
                )
                predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
                log.info("SAM2 loaded: %s (device=%s)", ckpt.name, device)
                return predictor
        log.warning("No SAM2 checkpoint found under %s", settings.models_dir / "sam2")
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load SAM2: %s", exc)
        return None


def _write_mask(path: Path, mask: np.ndarray) -> bool:
    """Write a single-channel uint8 mask; verify the file exists afterwards."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    mask = np.ascontiguousarray(mask.astype(np.uint8))
    ok = cv2.imwrite(str(path), mask)
    if ok and path.exists() and path.stat().st_size > 0:
        return True
    # OpenCV can return False on Windows for some array layouts — try PIL.
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(mask, mode="L").save(path)
        return path.exists() and path.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        log.warning("[tracking] failed to write mask %s: %s", path.name, exc)
        return False


def _rect_fallback(ctx, frames_dir, masks_dir, target_hw: tuple | None = None) -> int:
    """Write rectangle masks from ctx.initial_bbox. Returns number of files written."""
    x1, y1, x2, y2 = map(int, ctx.initial_bbox)
    written = 0
    for fp in sorted(frames_dir.glob("*.png")):
        if target_hw is not None:
            th, tw = target_hw
            mask = np.zeros((th, tw), dtype=np.uint8)
        else:
            img = cv2.imread(str(fp))
            if img is None:
                log.warning("[tracking] cannot read frame %s for rect fallback", fp.name)
                continue
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            th, tw = mask.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(tw, x2), min(th, y2)
        if x2c > x1c and y2c > y1c:
            mask[y1c:y2c, x1c:x2c] = 255
        if _write_mask(masks_dir / fp.name, mask):
            written += 1
    return written


def _jpeg_dir_for_sam2(frames_dir: Path) -> Path:
    """SAM 2's video predictor expects a folder of JPEGs named 0.jpg, 1.jpg, ...

    Our pipeline writes PNGs with 5-digit indices. We create a sibling
    folder with the SAM 2-expected layout (symlinks would be cleaner but
    don't always work on Windows, so we re-encode).
    """
    jpeg_dir = frames_dir.parent / "frames_jpeg"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    # cheap idempotency: only rebuild if counts differ
    pngs = sorted(frames_dir.glob("*.png"))
    if len(list(jpeg_dir.glob("*.jpg"))) != len(pngs):
        for jpg in jpeg_dir.glob("*.jpg"):
            jpg.unlink()
        for i, fp in enumerate(pngs):
            img = cv2.imread(str(fp))
            cv2.imwrite(str(jpeg_dir / f"{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jpeg_dir


def _to_2d_mask(mask_logits, target_hw: tuple) -> np.ndarray:
    """Normalize SAM 2's mask_logits to a HxW uint8 mask at the target resolution.

    SAM 2 returns logits at its internal resolution and the leading dims
    can be (N_objs, H, W) or (1, N_objs, H, W) depending on version, so
    we squeeze first and then resize.
    """
    import torch  # type: ignore

    if isinstance(mask_logits, torch.Tensor):
        arr = mask_logits.detach().cpu().numpy()
    else:
        arr = np.asarray(mask_logits)
    # squeeze leading singleton dims until we are 2D or 3D
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:  # (N_objs, H, W) -> take object 0
        arr = arr[0]
    mask = (arr > 0.0).astype(np.uint8) * 255
    th, tw = target_hw
    if mask.shape[:2] != (th, tw):
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    return mask


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    masks_dir: Path = ctx.run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    ctx.masks_dir = masks_dir

    frame_paths = sorted(ctx.frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(
            f"[tracking] No frames found in {ctx.frames_dir}. Ingest stage produced nothing."
        )
    target_hw = cv2.imread(str(frame_paths[0])).shape[:2]
    log.info("[tracking] read %d frames at %dx%d.", len(frame_paths), target_hw[1], target_hw[0])

    predictor = _load_sam2_video_predictor()
    if predictor is None:
        log.info("[tracking] Using rectangle-mask fallback (no SAM2 weights).")
        written = _rect_fallback(ctx, ctx.frames_dir, masks_dir, target_hw=target_hw)
        if written == 0:
            raise RuntimeError(f"[tracking] rectangle fallback wrote 0 files to {masks_dir}.")
        log.info("[tracking] wrote %d mask PNGs (fallback).", written)
        return

    try:
        import numpy as np  # noqa: F811
        import torch  # type: ignore

        jpeg_dir = _jpeg_dir_for_sam2(ctx.frames_dir)
        n_frames = len(frame_paths)

        with torch.inference_mode():
            state = predictor.init_state(video_path=str(jpeg_dir))
            predictor.reset_state(state)
            obj_id = 1
            x1, y1, x2, y2 = ctx.initial_bbox
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
            )

            masks_by_idx: dict[int, np.ndarray] = {}
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                mask = _to_2d_mask(mask_logits, target_hw)
                masks_by_idx[frame_idx] = mask

        log.info("[tracking] SAM2 returned masks for %d / %d frames.",
                 len(masks_by_idx), n_frames)

        written = 0
        for i, fp in enumerate(frame_paths):
            mask = masks_by_idx.get(i)
            if mask is None:
                mask = np.zeros(target_hw, dtype=np.uint8)
            if _write_mask(masks_dir / fp.name, mask):
                written += 1
            else:
                log.warning("[tracking] imwrite failed for frame %s (mask shape %s)", fp.name, mask.shape)

        # Fill any missing frames with rectangle masks.
        if written < n_frames:
            log.warning(
                "[tracking] SAM2 only wrote %d/%d masks; filling gaps with rectangle fallback.",
                written, n_frames,
            )
            for i, fp in enumerate(frame_paths):
                out = masks_dir / fp.name
                if out.exists() and out.stat().st_size > 0:
                    continue
                x1, y1, x2, y2 = map(int, ctx.initial_bbox)
                mask = np.zeros(target_hw, dtype=np.uint8)
                th, tw = target_hw
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(tw, x2), min(th, y2)
                if x2c > x1c and y2c > y1c:
                    mask[y1c:y2c, x1c:x2c] = 255
                _write_mask(out, mask)
    except Exception as exc:  # noqa: BLE001
        log.exception("[tracking] SAM2 inference failed; rectangle fallback: %s", exc)
        written = _rect_fallback(ctx, ctx.frames_dir, masks_dir, target_hw=target_hw)

    written = len([p for p in masks_dir.glob("*.png") if p.stat().st_size > 0])
    if written == 0:
        raise RuntimeError(
            f"[tracking] No masks were written to {masks_dir}. "
            "Both SAM 2 and rectangle fallback produced nothing -- check ctx.initial_bbox and disk permissions."
        )
    log.info("[tracking] wrote %d mask PNGs to %s.", written, masks_dir)
