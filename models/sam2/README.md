# SAM 2 weights

Place one (or more) of the following checkpoints here:

```
models/sam2/
├── sam2.1_hiera_tiny.pt        # ~150 MB  (fastest)
├── sam2.1_hiera_small.pt       # ~190 MB
├── sam2.1_hiera_base_plus.pt   # ~320 MB
└── sam2.1_hiera_large.pt       # ~900 MB  (best quality — DEFAULT)
```

Download (large variant, from project root):

```bash
curl -L -o models/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Install the Python package:

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Used by: `backend/pipeline/tracking.py` for promptable **video segmentation**:
the bbox from Grounding DINO seeds the first frame and SAM 2 propagates the
mask through every subsequent frame.

Paper: Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.

License: Apache 2.0.
