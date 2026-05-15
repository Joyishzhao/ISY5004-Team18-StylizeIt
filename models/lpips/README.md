# LPIPS weights

AlexNet LPIPS weights are bundled inside the `lpips` pip package:

```bash
pip install lpips
```

No file is required in this folder — it exists only as a placeholder so
the layout matches every other model dir.

Used by: `backend/evaluation/metrics.py` for the **Background LPIPS**
metric (we mask out the target region and compute LPIPS only on the
background, which is exactly what we want to preserve).

Paper: Zhang et al., "The Unreasonable Effectiveness of Deep Features as
a Perceptual Metric", CVPR 2018.
