"""Stage 4 - Diffusion-based stylization (target region only).

Implements three execution paths, picked in this order:

    1. ControlNet + SD-Inpainting  (best quality, requires ControlNet weights)
    2. SD-Inpainting               (good quality, just diffusers)
    3. OpenCV fallback             (no model weights, deterministic, runs on CPU)

Weights expected (auto-cached by diffusers on first run if absent):

    models/stable_diffusion/sd-inpainting/             (or HF: runwayml/stable-diffusion-inpainting)
    models/controlnet/canny/                           (or HF: lllyasviel/control_v11p_sd15_canny)

ControlNet is configured via configs/*.yaml `generation.controlnet.enabled`.

References:
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion
    Models." CVPR 2022. (Stable Diffusion)
    Zhang et al. "Adding Conditional Control to Text-to-Image Diffusion
    Models." ICCV 2023. (ControlNet)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.core.config import settings

log = logging.getLogger("stylizeit.generation")

# Module-level cache to avoid reloading 4GB of weights for every run.
_cache: dict = {"pipe": None, "kind": None}


# ---------------------------------------------------------------------------
# Pipeline loaders
# ---------------------------------------------------------------------------

def _hf_or_local(local_subdir: str, hf_repo_id: str) -> str:
    """Prefer local snapshot under models/, else fall back to HF Hub id."""
    local = settings.models_dir / local_subdir
    return str(local) if local.exists() and any(local.iterdir()) else hf_repo_id


def _load_controlnet_inpaint_pipe():
    """ControlNet (canny) + SD-1.5 Inpainting."""
    try:
        import torch  # type: ignore
        from diffusers import (  # type: ignore
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
        )

        dtype = torch.float16 if settings.device == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            _hf_or_local("controlnet/canny", "lllyasviel/control_v11p_sd15_canny"),
            torch_dtype=dtype,
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            _hf_or_local("stable_diffusion/sd-inpainting", "runwayml/stable-diffusion-inpainting"),
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,  # disable for offline batch runs
        )
        pipe.to(settings.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        log.info("Loaded ControlNet + SD-Inpaint pipeline.")
        return pipe
    except Exception as exc:  # noqa: BLE001
        log.warning("ControlNet pipeline unavailable: %s", exc)
        return None


def _load_inpaint_pipe():
    """Plain SD-1.5 Inpainting (no ControlNet)."""
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionInpaintPipeline  # type: ignore

        dtype = torch.float16 if settings.device == "cuda" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            _hf_or_local("stable_diffusion/sd-inpainting", "runwayml/stable-diffusion-inpainting"),
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.to(settings.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        log.info("Loaded SD-Inpaint pipeline (no ControlNet).")
        return pipe
    except Exception as exc:  # noqa: BLE001
        log.warning("SD-Inpaint pipeline unavailable: %s", exc)
        return None


def _pick_pipe(use_controlnet: bool):
    if _cache["pipe"] is not None:
        return _cache["pipe"], _cache["kind"]

    pipe = None
    kind = None
    if use_controlnet:
        pipe = _load_controlnet_inpaint_pipe()
        kind = "controlnet" if pipe is not None else None
    if pipe is None:
        pipe = _load_inpaint_pipe()
        kind = "inpaint" if pipe is not None else None

    _cache.update(pipe=pipe, kind=kind)
    return pipe, kind


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(config_name: str = "default.yaml") -> dict:
    """Read the YAML preset chosen by the user (best-effort)."""
    try:
        import yaml  # type: ignore

        path = settings.configs_dir / config_name
        if not path.exists():
            path = settings.configs_dir / "default.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read config preset: %s", exc)
    return {}


def _make_generator(seed: int):
    """Torch RNG with a fixed seed — same seed every frame reduces style drift."""
    import torch  # type: ignore

    device = settings.device if settings.device in ("cuda", "mps", "cpu") else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return torch.Generator(device=device).manual_seed(int(seed))


def _build_chained_init(
    frame_bgr: np.ndarray,
    mask_gray: np.ndarray,
    prev_stylized: np.ndarray | None,
    blend: float,
) -> np.ndarray:
    """Use the previous stylized frame inside the mask as the inpaint init.

    This is the main lever for temporal consistency *before* RAFT: each
    frame's diffusion starts closer to the last frame's look.
    """
    if prev_stylized is None or blend <= 0:
        return frame_bgr
    if prev_stylized.shape[:2] != frame_bgr.shape[:2]:
        prev_stylized = cv2.resize(
            prev_stylized, (frame_bgr.shape[1], frame_bgr.shape[0])
        )
    m = (mask_gray > 127)[..., None].astype(np.float32)
    blended = blend * prev_stylized.astype(np.float32) + (1.0 - blend) * frame_bgr.astype(np.float32)
    return (blended * m + frame_bgr.astype(np.float32) * (1.0 - m)).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fallback (no diffusion weights)
# ---------------------------------------------------------------------------

def _fallback_stylize(frame_bgr: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    quant = (smooth // 32) * 32  # color quantisation -> cartoonish
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(quant, edges_bgr)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(ctx) -> None:  # type: ignore[no-untyped-def]
    stylized_dir: Path = ctx.run_dir / "stylized"
    stylized_dir.mkdir(parents=True, exist_ok=True)
    ctx.stylized_dir = stylized_dir

    cfg = _load_yaml_config(getattr(ctx, "config_name", "default.yaml"))
    gen_cfg = cfg.get("generation", {})
    cn_cfg = gen_cfg.get("controlnet", {}) or {}
    tc_cfg = gen_cfg.get("temporal_consistency", {}) or {}
    use_controlnet = bool(cn_cfg.get("enabled", False))
    num_steps = int(gen_cfg.get("num_inference_steps", 25))
    guidance = float(gen_cfg.get("guidance_scale", 7.5))
    strength = float(gen_cfg.get("strength", 0.72))
    cn_scale = float(cn_cfg.get("conditioning_scale", 0.6))
    seed = int(gen_cfg.get("seed", 42))
    chain_enabled = bool(tc_cfg.get("enabled", True))
    chain_blend = float(tc_cfg.get("init_blend", 0.55))
    prompt_suffix = gen_cfg.get(
        "style_prompt_suffix",
        "anime style, vibrant colors, clean lineart, highly detailed, studio ghibli aesthetic",
    )
    negative_prompt = gen_cfg.get(
        "negative_prompt",
        "blurry, low quality, deformed, extra limbs, inconsistent style, flickering",
    )

    pipe, kind = _pick_pipe(use_controlnet)
    style_prompt = f"{ctx.prompt}, {prompt_suffix}"
    generator = _make_generator(seed) if kind else None
    log.info(
        "[generation] temporal_consistency=%s init_blend=%.2f seed=%d strength=%.2f steps=%d",
        chain_enabled, chain_blend, seed, strength, num_steps,
    )

    frame_paths = sorted(ctx.frames_dir.glob("*.png"))
    mask_paths = sorted(ctx.masks_dir.glob("*.png"))
    log.info(
        "[generation] read %d frames and %d masks. pipe=%s use_controlnet=%s",
        len(frame_paths), len(mask_paths), kind or "fallback", use_controlnet,
    )

    if not frame_paths:
        raise RuntimeError(
            f"[generation] No frames in {ctx.frames_dir} -- ingest stage failed silently."
        )
    if not mask_paths:
        raise RuntimeError(
            f"[generation] No masks in {ctx.masks_dir} -- tracking stage produced nothing."
        )
    if len(frame_paths) != len(mask_paths):
        log.warning(
            "[generation] frame/mask count mismatch (%d vs %d); will zip to the shorter.",
            len(frame_paths), len(mask_paths),
        )

    written = 0
    prev_stylized: np.ndarray | None = None
    for i, (fp, mp) in enumerate(zip(frame_paths, mask_paths)):
        frame = cv2.imread(str(fp))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            log.warning("[generation] cv2.imread failed for frame %d (%s / %s)", i, fp, mp)
            continue
        # Defensive: align mask to frame size if SAM 2 returned a different
        # resolution upstream and tracking forgot to resize.
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(
                mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        init_frame = (
            _build_chained_init(frame, mask, prev_stylized, chain_blend)
            if chain_enabled
            else frame
        )

        try:
            if pipe is None:
                stylized = _fallback_stylize(frame)
            else:
                stylized = _diffusion_inpaint(
                    pipe,
                    kind,
                    init_frame,
                    mask,
                    style_prompt,
                    negative_prompt,
                    num_steps,
                    guidance,
                    strength=strength,
                    controlnet_scale=cn_scale,
                    generator=generator,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("[generation] frame %d failed (%s); using fallback stylizer.", i, exc)
            stylized = _fallback_stylize(frame)

        cv2.imwrite(str(stylized_dir / fp.name), stylized)
        prev_stylized = stylized
        written += 1
        if (i + 1) % 20 == 0:
            log.info("[generation] %d/%d frames done.", i + 1, len(frame_paths))

    if written == 0:
        raise RuntimeError(
            f"[generation] Wrote 0 frames to {stylized_dir}. "
            "Possible causes: zip yielded nothing (mask count = 0?) "
            "or cv2.imread failed on every frame."
        )
    log.info("[generation] wrote %d stylized PNGs (kind=%s).", written, kind or "fallback")


def _diffusion_inpaint(
    pipe,
    kind: str,
    frame_bgr: np.ndarray,
    mask_gray: np.ndarray,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance: float,
    *,
    strength: float = 0.72,
    controlnet_scale: float = 0.6,
    generator=None,
) -> np.ndarray:
    from PIL import Image  # type: ignore

    h, w = frame_bgr.shape[:2]
    # SD-1.5 family works best at 512 multiples; we round to nearest 64.
    tw = max(64, (w // 64) * 64)
    th = max(64, (h // 64) * 64)

    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).resize((tw, th))
    pil_mask = Image.fromarray(mask_gray).resize((tw, th))

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        mask_image=pil_mask,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=generator,
    )
    # Lower strength = keep more of the init image (chained prev frame) -> less flicker.
    try:
        kwargs["strength"] = max(0.35, min(1.0, strength))
    except TypeError:
        pass

    if kind == "controlnet":
        edges = cv2.Canny(frame_bgr, 80, 160)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(edges_rgb).resize((tw, th))
        kwargs["control_image"] = control_image
        kwargs["controlnet_conditioning_scale"] = controlnet_scale

    try:
        out = pipe(**kwargs).images[0]
    except TypeError:
        kwargs.pop("strength", None)
        out = pipe(**kwargs).images[0]
    out = out.resize((w, h))
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
