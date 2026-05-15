# Stable Diffusion (Inpainting)

Place the diffusers snapshot here:

```
models/stable_diffusion/
└── sd-inpainting/             # full HF repo snapshot
    ├── model_index.json
    ├── unet/
    ├── vae/
    ├── text_encoder/
    └── ...
```

Download (recommended):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    local_dir="models/stable_diffusion/sd-inpainting",
    local_dir_use_symlinks=False,
)
PY
```

Alternatives:

* **SDXL Inpainting** (higher fidelity, needs ~16 GB VRAM):
  `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
* **AnimateDiff** (better motion coherence at the cost of speed):
  `guoyww/animatediff-motion-adapter-v1-5-2`

Used by: `backend/pipeline/generation.py` for **target-only stylization**.
The mask from SAM 2 confines the diffusion edit to the subject so the
background is left bit-exact.

Paper: Rombach et al., "High-Resolution Image Synthesis with Latent
Diffusion Models", CVPR 2022.

License: CreativeML Open RAIL-M — see the model card on Hugging Face.
