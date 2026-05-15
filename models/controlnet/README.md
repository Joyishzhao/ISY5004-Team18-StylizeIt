# ControlNet weights (optional)

ControlNet conditions diffusion inpainting on a structural map (edges,
depth, or pose) so the anime version preserves the **shape and motion**
of the real subject. Strongly recommended for "person" prompts.

Suggested layout:

```
models/controlnet/
├── canny/         # control_v11p_sd15_canny
├── depth/         # control_v11f1p_sd15_depth
└── openpose/      # control_v11p_sd15_openpose
```

Download (canny example):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lllyasviel/control_v11p_sd15_canny",
    local_dir="models/controlnet/canny",
    local_dir_use_symlinks=False,
)
PY
```

Used by: `backend/pipeline/generation.py` (optional code path) to feed
edges / pose / depth into the diffusion pass.

Paper: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion
Models", ICCV 2023.

License: OpenRAIL.
