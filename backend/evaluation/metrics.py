"""Stage 8 - Evaluation metrics.

Implements the four metric families called out in doc/PRD.md §6:

    * mIoU & Boundary F-Measure   - tracking quality (DAVIS-style)
    * Warping Error               - temporal stability (RAFT)
    * CLIP Score                  - generation / prompt alignment
    * Background LPIPS            - background preservation

For ground-truth IoU/F-measure you need a labelled video (e.g. DAVIS); when
ground truth is unavailable we still compute warping error / CLIP / LPIPS
which are reference-free w.r.t. the input.

Heavy model deps (CLIP, LPIPS) are imported lazily to keep the pipeline
working when the weights are missing.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("stylizeit.eval")


def _warping_error(frames_dir, stylized_dir) -> float:
    f = sorted(frames_dir.glob("*.png"))
    s = sorted(stylized_dir.glob("*.png"))
    if len(f) < 2 or len(s) < 2:
        return 0.0
    errs = []
    for i in range(len(f) - 1):
        prev = cv2.cvtColor(cv2.imread(str(f[i])), cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(cv2.imread(str(f[i + 1])), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (xs + flow[..., 0]).astype(np.float32)
        map_y = (ys + flow[..., 1]).astype(np.float32)
        sty_prev = cv2.imread(str(s[i])).astype(np.float32)
        sty_curr = cv2.imread(str(s[i + 1])).astype(np.float32)
        warped = cv2.remap(sty_prev, map_x, map_y, cv2.INTER_LINEAR)
        errs.append(float(np.mean(np.abs(warped - sty_curr)) / 255.0))
    return float(np.mean(errs))


def _clip_score(stylized_dir, prompt: str) -> float:
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        scores = []
        for fp in sorted(stylized_dir.glob("*.png"))[::5]:
            image = Image.open(fp).convert("RGB")
            inputs = proc(text=[prompt], images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
            scores.append(float(logits.softmax(dim=-1)[0, 0].item()))
        return float(np.mean(scores)) if scores else 0.0
    except Exception as exc:  # noqa: BLE001
        log.warning("CLIP score skipped: %s", exc)
        return 0.0


def _background_lpips(frames_dir, stylized_dir, masks_dir) -> float:
    try:
        import lpips  # type: ignore
        import torch  # type: ignore

        loss_fn = lpips.LPIPS(net="alex")
        loss_fn.eval()
        f = sorted(frames_dir.glob("*.png"))
        s = sorted(stylized_dir.glob("*.png"))
        m = sorted(masks_dir.glob("*.png"))
        vals = []
        for fp, sp, mp in zip(f, s, m):
            orig = cv2.imread(str(fp))
            sty = cv2.imread(str(sp))
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            inv = (mask < 128)[..., None].astype(np.float32)
            a = (orig.astype(np.float32) * inv).transpose(2, 0, 1) / 127.5 - 1
            b = (sty.astype(np.float32) * inv).transpose(2, 0, 1) / 127.5 - 1
            with torch.no_grad():
                d = loss_fn(torch.tensor(a)[None], torch.tensor(b)[None])
            vals.append(float(d.item()))
        return float(np.mean(vals)) if vals else 0.0
    except Exception as exc:  # noqa: BLE001
        log.warning("Background LPIPS skipped: %s", exc)
        return 0.0


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    ctx.metrics = {
        "miou": None,           # populated by scripts/run_eval.py vs DAVIS GT
        "boundary_f": None,     # ditto
        "warping_error": _warping_error(ctx.frames_dir, ctx.stylized_dir),
        "clip_score": _clip_score(ctx.stylized_dir, ctx.prompt),
        "background_lpips": _background_lpips(
            ctx.frames_dir, ctx.stylized_dir, ctx.masks_dir
        ),
    }
