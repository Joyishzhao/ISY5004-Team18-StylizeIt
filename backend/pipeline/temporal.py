"""Stage 5 - Temporal stabilization via RAFT optical flow warping.

This stage is the project's headline contribution: kill the per-frame
flicker that all naive video-by-diffusion methods suffer from.

Approach (high level):
    For every consecutive pair (t-1, t) of ORIGINAL frames:
        1. flow_fwd = RAFT(frame_{t-1}, frame_t)          [forward optical flow]
        2. flow_bwd = RAFT(frame_t, frame_{t-1})          [backward optical flow]
        3. warp stable_{t-1} along flow_fwd  -> warped_prev_stable
        4. occlusion_mask = (||flow_fwd + warp(flow_bwd)|| > thr)
           (i.e. pixels whose round-trip doesn't return -> just appeared)
        5. inside the target mask AND outside occlusion:
              stable_t = α * stylized_t + (1-α) * warped_prev_stable
           otherwise:
              stable_t = stylized_t                       [trust current frame]

If RAFT cannot be imported / loaded, we fall back to the simple EMA blend
inside the mask. The pipeline contract (writing PNGs into ./stable/) is
identical so nothing downstream cares.

Reference:
    Teed & Deng. "RAFT: Recurrent All-Pairs Field Transforms for Optical
    Flow." ECCV 2020.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.core.config import settings

log = logging.getLogger("stylizeit.temporal")


# ---------------------------------------------------------------------------
# RAFT loader (uses torchvision built-in weights -> no manual download)
# ---------------------------------------------------------------------------

_cache: dict = {"model": None, "device": None, "transforms": None}


def _load_raft():
    """Lazy-load torchvision's RAFT-Large with the SKHT-V2 pretrained weights."""
    if _cache["model"] is not None:
        return _cache["model"], _cache["device"], _cache["transforms"]
    try:
        import torch  # type: ignore
        from torchvision.models.optical_flow import (  # type: ignore
            Raft_Large_Weights,
            raft_large,
        )

        weights = Raft_Large_Weights.C_T_SKHT_V2
        transforms = weights.transforms()
        device = (
            "cuda" if (settings.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        model = raft_large(weights=weights, progress=True).to(device).eval()
        _cache.update(model=model, device=device, transforms=transforms)
        log.info("RAFT loaded (torchvision raft_large, device=%s).", device)
        return model, device, transforms
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load RAFT: %s — temporal stage will fall back to EMA.", exc)
        return None, None, None


# ---------------------------------------------------------------------------
# Flow utilities
# ---------------------------------------------------------------------------

def _to_tensor_batch(img_bgr: np.ndarray, device: str):
    """Convert HxWx3 BGR uint8 to a 1x3xHxW float tensor (RGB) on device."""
    import torch  # type: ignore

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def _compute_flow(model, transforms, device, img_a_bgr: np.ndarray, img_b_bgr: np.ndarray) -> np.ndarray:
    """Return HxWx2 float32 flow field (in pixel units) from a->b."""
    import torch  # type: ignore

    h, w = img_a_bgr.shape[:2]
    # RAFT prefers dimensions that are multiples of 8.
    nh = (h // 8) * 8
    nw = (w // 8) * 8
    a = cv2.resize(img_a_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    b = cv2.resize(img_b_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    ta = _to_tensor_batch(a, device)
    tb = _to_tensor_batch(b, device)
    ta_t, tb_t = transforms(ta, tb)

    with torch.no_grad():
        flow_list = model(ta_t, tb_t)  # list of refined flow predictions
        flow = flow_list[-1][0].cpu().numpy()  # 2 x nh x nw
    flow = flow.transpose(1, 2, 0)  # nh, nw, 2

    # rescale to original resolution
    if (nh, nw) != (h, w):
        flow_resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_resized[..., 0] *= w / nw
        flow_resized[..., 1] *= h / nh
        flow = flow_resized
    return flow.astype(np.float32)


def _warp_image(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp *img* (frame t-1) onto the grid of frame t.

    torchvision RAFT returns forward flow from image_a -> image_b:
    a pixel at (x, y) in frame_a moves to (x + dx, y + dy) in frame_b.

    cv2.remap samples the SOURCE at (map_x, map_y) for each OUTPUT pixel,
    so we need backward mapping: output(x, y) <- source(x - dx, y - dy).
    (Using + instead of - was a bug that made temporal blending worse.)
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xs - flow[..., 0]).astype(np.float32)
    map_y = (ys - flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, thresh: float = 1.5) -> np.ndarray:
    """Build an occlusion mask via forward-backward consistency.

    A pixel is considered NEWLY VISIBLE (occluded in previous frame) when
    round-trip flow disagrees by more than `thresh` pixels.

    Returns a HxW uint8 mask: 255 = occluded (don't trust previous frame),
    0 = consistent (safe to blend).
    """
    h, w = flow_fwd.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = xs + flow_fwd[..., 0]
    fy = ys + flow_fwd[..., 1]
    # sample backward flow at (fx, fy)
    bwd_x = cv2.remap(flow_bwd[..., 0], fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bwd_y = cv2.remap(flow_bwd[..., 1], fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    diff_x = flow_fwd[..., 0] + bwd_x
    diff_y = flow_fwd[..., 1] + bwd_y
    err = np.sqrt(diff_x ** 2 + diff_y ** 2)
    return (err > thresh).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Stabilizers
# ---------------------------------------------------------------------------

def _ema_blend(prev: np.ndarray, curr: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    m = (mask.astype(np.float32) / 255.0)[..., None]
    blended = alpha * curr.astype(np.float32) + (1 - alpha) * prev.astype(np.float32)
    return (blended * m + curr.astype(np.float32) * (1 - m)).clip(0, 255).astype(np.uint8)


def _raft_warp_blend(
    prev_stable: np.ndarray,
    curr_stylized: np.ndarray,
    prev_orig: np.ndarray,
    curr_orig: np.ndarray,
    mask: np.ndarray,
    model,
    transforms,
    device,
    alpha: float = 0.28,
    occ_thresh: float = 1.5,
) -> np.ndarray:
    """RAFT-warped stable_{t-1} mixed with stylized_t inside the target mask."""

    flow_fwd = _compute_flow(model, transforms, device, prev_orig, curr_orig)
    flow_bwd = _compute_flow(model, transforms, device, curr_orig, prev_orig)
    occ = _occlusion_mask(flow_fwd, flow_bwd, thresh=occ_thresh)

    warped_prev = _warp_image(prev_stable, flow_fwd)

    # blend weight w(x):
    #   - inside target mask AND consistent flow -> α  (use mostly previous)
    #   - inside target mask AND occluded        -> 1  (trust current)
    #   - outside target mask                    -> 1  (we only stabilize the subject)
    target = (mask > 127).astype(np.float32)
    consistent = (occ == 0).astype(np.float32)
    blend_prev = target * consistent * (1.0 - alpha)         # weight for warped_prev
    blend_curr = 1.0 - blend_prev                            # weight for curr_stylized
    blend_prev = blend_prev[..., None]
    blend_curr = blend_curr[..., None]

    stable = (
        warped_prev.astype(np.float32) * blend_prev
        + curr_stylized.astype(np.float32) * blend_curr
    )
    return stable.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage entry
# ---------------------------------------------------------------------------

def _load_yaml() -> dict:
    try:
        import yaml  # type: ignore

        path = settings.configs_dir / "default.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read config preset: %s", exc)
    return {}


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    stable_dir: Path = ctx.run_dir / "stable"
    stable_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml().get("temporal", {}) or {}
    method = cfg.get("method", "optical_flow_warp")
    ema_alpha = float(cfg.get("ema_alpha", 0.6))
    raft_alpha = float(cfg.get("raft_alpha", 0.28))
    occ_thresh = float(cfg.get("occlusion_thresh", 1.5))

    frame_paths = sorted(ctx.frames_dir.glob("*.png"))
    stylized_paths = sorted(ctx.stylized_dir.glob("*.png"))
    mask_paths = sorted(ctx.masks_dir.glob("*.png"))
    log.info(
        "[temporal] read frames=%d stylized=%d masks=%d (method=%s)",
        len(frame_paths), len(stylized_paths), len(mask_paths), method,
    )
    if not stylized_paths:
        raise RuntimeError(
            f"[temporal] {ctx.stylized_dir} has no PNGs -- generation stage produced nothing."
        )

    model = transforms = device = None
    if method == "optical_flow_warp":
        model, device, transforms = _load_raft()
    if model is None and method == "optical_flow_warp":
        log.warning("[temporal] RAFT unavailable, falling back to EMA.")
        method = "ema"

    prev_stable: Optional[np.ndarray] = None
    prev_orig: Optional[np.ndarray] = None
    n = len(frame_paths)
    written = 0

    for i, (fp, sp, mp) in enumerate(zip(frame_paths, stylized_paths, mask_paths)):
        orig = cv2.imread(str(fp))
        stylized = cv2.imread(str(sp))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if stylized is None or orig is None:
            log.warning("[temporal] cv2.imread failed at frame %d (%s, %s)", i, fp, sp)
            continue
        if mask is None:
            mask = np.zeros(orig.shape[:2], dtype=np.uint8)
        if mask.shape[:2] != orig.shape[:2]:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
        if stylized.shape[:2] != orig.shape[:2]:
            stylized = cv2.resize(stylized, (orig.shape[1], orig.shape[0]))

        try:
            if prev_stable is None:
                stable = stylized
            elif method == "optical_flow_warp":
                stable = _raft_warp_blend(
                    prev_stable=prev_stable,
                    curr_stylized=stylized,
                    prev_orig=prev_orig,
                    curr_orig=orig,
                    mask=mask,
                    model=model,
                    transforms=transforms,
                    device=device,
                    alpha=raft_alpha,
                    occ_thresh=occ_thresh,
                )
            else:
                stable = _ema_blend(prev_stable, stylized, mask, alpha=ema_alpha)
        except Exception as exc:  # noqa: BLE001
            log.exception("[temporal] frame %d stabilisation failed (%s); using stylized as-is.", i, exc)
            stable = stylized

        cv2.imwrite(str(stable_dir / fp.name), stable)
        prev_stable = stable
        prev_orig = orig
        written += 1

        if (i + 1) % 20 == 0:
            log.info("[temporal] %d/%d frames done (method=%s)", i + 1, n, method)

    if written == 0:
        raise RuntimeError(
            f"[temporal] Wrote 0 files to {stable_dir}. Upstream dirs were inconsistent."
        )
    log.info("[temporal] wrote %d stable PNGs (method=%s).", written, method)
    ctx.stylized_dir = stable_dir
