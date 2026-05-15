"""Stage 2 - Grounding (text -> bounding box).

Uses Grounding DINO via the `transformers` library
(`IDEA-Research/grounding-dino-tiny`). Going through transformers instead
of the upstream `groundingdino-py` pip package avoids a Windows-only
`UnicodeDecodeError: gbk` failure in that package's setup.py and is
otherwise functionally identical.

If model loading fails (no internet on first run / no GPU memory / etc.)
the stage falls back to a centred bbox so downstream stages still work
and the API contract holds.

Reference:
    Liu et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training
    for Open-Set Object Detection." arXiv:2303.05499.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from backend.core.config import settings

log = logging.getLogger("stylizeit.grounding")

_GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_cached = {"model": None, "processor": None, "device": None}


def _load_grounding_dino():
    """Lazy-load and cache the HF Grounding DINO pipeline."""
    if _cached["model"] is not None:
        return _cached["model"], _cached["processor"], _cached["device"]
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        device = (
            settings.device
            if (settings.device == "cuda" and torch.cuda.is_available())
            or settings.device in ("mps", "cpu")
            else "cpu"
        )
        processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            _GROUNDING_DINO_MODEL_ID
        ).to(device)
        model.eval()
        _cached.update(model=model, processor=processor, device=device)
        return model, processor, device
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load Grounding DINO via transformers: %s", exc)
        return None, None, None


def _normalize_prompt(prompt: str) -> str:
    """Grounding DINO expects lowercase, period-terminated noun phrases."""
    p = prompt.strip().lower()
    if not p.endswith("."):
        p = p + "."
    return p


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    first_frame_path: Path = sorted(ctx.frames_dir.glob("*.png"))[0]
    img = cv2.imread(str(first_frame_path))
    h, w = img.shape[:2]

    model, processor, device = _load_grounding_dino()
    if model is None:
        log.info("Using fallback centered bbox (no Grounding DINO).")
        ctx.initial_bbox = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
        return

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        image = Image.open(first_frame_path).convert("RGB")
        text = _normalize_prompt(ctx.prompt)

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # post-process to image coordinates (xyxy in pixels)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            box_threshold=0.30,
            text_threshold=0.25,
            target_sizes=[(h, w)],
        )[0]

        if results["boxes"].shape[0] == 0:
            log.warning("Grounding DINO found no box for prompt: %r", text)
            ctx.initial_bbox = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
            return

        # take the highest-scoring detection
        scores = results["scores"].detach().cpu()
        idx = int(scores.argmax())
        x1, y1, x2, y2 = results["boxes"][idx].tolist()
        # clamp & store
        ctx.initial_bbox = [
            max(0.0, x1),
            max(0.0, y1),
            min(float(w), x2),
            min(float(h), y2),
        ]
        log.info("Grounding DINO bbox=%s score=%.3f", ctx.initial_bbox, float(scores[idx]))
    except Exception as exc:  # noqa: BLE001
        log.exception("Grounding DINO inference failed: %s", exc)
        ctx.initial_bbox = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
