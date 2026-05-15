"""Offline evaluator for tracking quality on DAVIS 2017.

Reads predicted masks under <run_dir>/masks/ and compares them to the
ground-truth annotations under data/davis2017/Annotations/<sequence>/.

Usage:
    python scripts/run_eval.py --run-dir artifacts/runs/run_xxx \
                               --gt-dir   data/davis2017/Annotations/bear
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 127
    gt_b = gt > 0
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(inter / union) if union > 0 else 0.0


def boundary_f(pred: np.ndarray, gt: np.ndarray, dilate: int = 3) -> float:
    pred_b = (pred > 127).astype(np.uint8)
    gt_b = (gt > 0).astype(np.uint8)
    pe = pred_b - cv2.erode(pred_b, np.ones((dilate, dilate), np.uint8))
    ge = gt_b - cv2.erode(gt_b, np.ones((dilate, dilate), np.uint8))
    inter = np.logical_and(pe, ge).sum()
    precision = inter / max(pe.sum(), 1)
    recall = inter / max(ge.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    args = ap.parse_args()

    pred_dir = args.run_dir / "masks"
    pred_paths = sorted(pred_dir.glob("*.png"))
    gt_paths = sorted(args.gt_dir.glob("*.png"))
    if not pred_paths or not gt_paths:
        raise SystemExit("Missing pred or GT masks.")

    n = min(len(pred_paths), len(gt_paths))
    ious, fs = [], []
    for i in range(n):
        p = cv2.imread(str(pred_paths[i]), cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(str(gt_paths[i]), cv2.IMREAD_GRAYSCALE)
        if p.shape != g.shape:
            g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(iou(p, g))
        fs.append(boundary_f(p, g))

    out = {
        "miou": float(np.mean(ious)),
        "boundary_f": float(np.mean(fs)),
        "n_frames": n,
    }
    print(json.dumps(out, indent=2))

    (args.run_dir / "metrics_offline.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
