# Grounding DINO weights

Place the following two files in this folder:

```
models/grounding_dino/
├── groundingdino_swint_ogc.pth        # ~700 MB checkpoint
└── GroundingDINO_SwinT_OGC.py         # model config
```

Download (from project root):

```bash
curl -L -o models/grounding_dino/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

curl -L -o models/grounding_dino/GroundingDINO_SwinT_OGC.py \
  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

pip install groundingdino-py
```

Used by: `backend/pipeline/grounding.py` to convert the user's text prompt
into an initial bounding box on the first frame of the video.

Paper: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training
for Open-Set Object Detection", arXiv:2303.05499.

License: Apache 2.0 (research & commercial use allowed; please cite the paper).
