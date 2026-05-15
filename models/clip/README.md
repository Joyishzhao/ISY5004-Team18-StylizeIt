# CLIP weights

We use OpenAI CLIP **ViT-B/32** as the default for CLIP Score evaluation.

Weights are downloaded **automatically** by `transformers` the first time
you instantiate the model:

```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

If you want a fully offline setup, snapshot the repo here:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="models/clip/vit-b32",
    local_dir_use_symlinks=False,
)
PY
```

Used by: `backend/evaluation/metrics.py` to compute the **CLIP Score**
(generation/prompt alignment) reported on the result page.

Paper: Radford et al., "Learning Transferable Visual Models From Natural
Language Supervision" (CLIP), ICML 2021.
