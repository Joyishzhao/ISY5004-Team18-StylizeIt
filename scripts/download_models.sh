#!/usr/bin/env bash
# Download all model weights into ./models/
# Usage:   bash scripts/download_models.sh
# Re-running is safe — existing files are skipped.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"/{grounding_dino,sam2,stable_diffusion,controlnet,raft,clip,lpips}

echo "[1/5] Grounding DINO"
if [ ! -f "$MODELS/grounding_dino/groundingdino_swint_ogc.pth" ]; then
  curl -L --fail -o "$MODELS/grounding_dino/groundingdino_swint_ogc.pth" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi
if [ ! -f "$MODELS/grounding_dino/GroundingDINO_SwinT_OGC.py" ]; then
  curl -L --fail -o "$MODELS/grounding_dino/GroundingDINO_SwinT_OGC.py" \
    https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
fi

echo "[2/5] SAM 2 (hiera_large)"
if [ ! -f "$MODELS/sam2/sam2.1_hiera_large.pt" ]; then
  curl -L --fail -o "$MODELS/sam2/sam2.1_hiera_large.pt" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

echo "[3/5] Stable Diffusion Inpainting"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    local_dir="$MODELS/stable_diffusion/sd-inpainting",
    local_dir_use_symlinks=False,
)
PY

echo "[4/5] ControlNet (canny)"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lllyasviel/control_v11p_sd15_canny",
    local_dir="$MODELS/controlnet/canny",
    local_dir_use_symlinks=False,
)
PY

echo "[5/5] RAFT (Sintel)"
if [ ! -f "$MODELS/raft/raft-things.pth" ]; then
  curl -L --fail -o "$MODELS/raft/raft-things.pth" \
    https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth || \
    echo "  (skipped: RAFT release missing — torchvision RAFT will be used at runtime)"
fi

echo
echo "Done. Model tree:"
find "$MODELS" -maxdepth 2 -type f | sort
