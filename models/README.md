# Models Directory

This folder holds the **pretrained weights** that the StylizeIt pipeline
calls during inference. We deliberately keep weights **out of git** (see
the project-level `.gitignore`) because they are large and licensed
separately.

## What goes where

| Stage | Sub-folder | Model | Why we use it |
| --- | --- | --- | --- |
| 2. Grounding | `grounding_dino/` | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Converts the user's **text prompt** into an initial bounding box on the first frame. Zero-shot, open-vocabulary. |
| 3. Tracking + Segmentation | `sam2/` | [SAM 2](https://github.com/facebookresearch/sam2) | Promptable segmentation **in video**: takes the box and propagates a precise mask through every frame. SOTA (Meta AI, 2024). |
| 4. Stylization | `stable_diffusion/` | [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) / [SDXL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | Text-guided diffusion **inpainting** restricted to the mask -> stylizes the target while preserving the rest. |
| 4b. Structure control | `controlnet/` | [ControlNet](https://huggingface.co/lllyasviel) (canny / openpose / depth) | Keeps the **shape and pose** of the original subject so the anime version aligns with the real motion. |
| 5. Temporal stability | `raft/` | [RAFT](https://github.com/princeton-vl/RAFT) | Dense optical flow between consecutive frames -> warping constraint that kills flicker. |
| 8. Evaluation | `clip/` | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | CLIP Score (prompt-image alignment) for generation quality. |
| 8. Evaluation | `lpips/` | [LPIPS](https://github.com/richzhang/PerceptualSimilarity) AlexNet weights | Perceptual distance on the **background** -> background preservation metric. |

## Installation - the easy path (Windows / Linux / macOS)

The repo ships two helper scripts in `scripts/`:

```powershell
# Windows PowerShell, from project root
.\scripts\download_models.ps1
```

```bash
# Linux / macOS / WSL, from project root
bash scripts/download_models.sh
```

Both scripts will:

1. Create every sub-folder below `models/`.
2. Pull weights from Hugging Face Hub & official release pages.
3. Verify file sizes and print a summary table.

You need `git`, `curl` (or `wget`), and a working `pip` environment first
(see the top-level README for the conda / venv setup).

## Installation - manual (per model)

Run these commands **from project root** so the relative paths match
what `backend/pipeline/*.py` expects.

### 1) Grounding DINO

```bash
mkdir -p models/grounding_dino
# weights (~700 MB)
curl -L -o models/grounding_dino/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# config
curl -L -o models/grounding_dino/GroundingDINO_SwinT_OGC.py \
  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
# Python package
pip install groundingdino-py
```

### 2) SAM 2

```bash
mkdir -p models/sam2
# weights (~900 MB for hiera_large)
curl -L -o models/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
# Python package
pip install "git+https://github.com/facebookresearch/sam2.git"
```

### 3) Stable Diffusion Inpainting (via diffusers)

```bash
mkdir -p models/stable_diffusion
# Option A: download whole snapshot (recommended for offline runs)
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    local_dir="models/stable_diffusion/sd-inpainting",
    local_dir_use_symlinks=False,
)
PY
# Option B: just `diffusers` cache (no manual download)
pip install diffusers transformers accelerate safetensors
```

### 4) ControlNet (optional but recommended)

```bash
mkdir -p models/controlnet
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lllyasviel/control_v11p_sd15_canny",
    local_dir="models/controlnet/canny",
    local_dir_use_symlinks=False,
)
PY
```

### 5) RAFT (optical flow)

```bash
mkdir -p models/raft
# pretrained on Sintel
curl -L -o models/raft/raft-sintel.pth \
  https://drive.usercontent.google.com/download?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM\&confirm=t
# or use torchvision's built-in RAFT (no manual download)
pip install "torchvision>=0.13"
```

### 6) CLIP & LPIPS (evaluation)

```bash
pip install transformers lpips
# CLIP weights download automatically the first time you use the model.
# LPIPS AlexNet weights are bundled inside the pip package.
```

## How the pipeline finds these files

Every pipeline stage under `backend/pipeline/*.py` looks for its weights
**relative to `models/`** through `backend.core.config.settings.models_dir`.
You can override that root with:

```bash
export STYLIZEIT_MODELS_DIR=/data/big_disk/stylizeit_models   # Linux / WSL
$env:STYLIZEIT_MODELS_DIR = "D:\\stylizeit_models"            # PowerShell
```

If a weight file is **missing**, the corresponding stage silently falls
back to a deterministic placeholder (rectangle mask / OpenCV stylizer /
no-op warp). That way you can develop the API, frontend, and metrics
without a GPU.

## Hardware notes

| Component | Minimum | Recommended |
| --- | --- | --- |
| GPU VRAM | 8 GB (SD-1.5 inpaint at 512x512) | 16 GB+ (SDXL inpaint or batch inference) |
| Disk | ~7 GB for all weights | ~20 GB if you cache multiple ControlNets |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 13+ | Ubuntu 22.04 with CUDA 12.x |
