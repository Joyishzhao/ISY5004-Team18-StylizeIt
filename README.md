# StylizeIt — Target-Level Video Stylization

> **ISY5004 Team 18** &mdash; Zhao Ziyang, Zhao Yuan, Jin Ziping
>
> A practical multimodal system that performs **target-level style editing** on
> videos. Unlike traditional global style transfer, StylizeIt achieves
> precise, semantic-based stylization of **one specific object** in the
> video while strictly preserving the background. Our v1 focuses on the
> *anime* style, but the framework is style-agnostic by design.

---

## 1. Why this project?

| Existing tool | What it lacks |
| --- | --- |
| Runway Aleph / Gen-2 (video-to-video) | Global style transfer — changes the whole scene. |
| Adobe After Effects (rotobrush) | Requires frame-by-frame manual rotoscoping. |
| TikTok filters | Fixed presets, no text-driven control, no object selectivity. |

**StylizeIt** combines three SOTA building blocks to fix all three gaps:

1. **Zero-shot text control** — type *"Turn the man in red into anime style"*; no manual masks.
2. **Target-specific precision** — only the subject is repainted; the background is bit-exact.
3. **Temporal stability** — optical-flow warping kills the flicker every per-frame diffusion editor suffers from.

## 2. Workflow at a glance

```
                                ┌──────────────────────────┐
                                │ User uploads video.mp4   │
                                │ + prompt (string)        │
                                └────────────┬─────────────┘
                                             │
                              POST /api/v1/runs (multipart)
                                             │
                                             ▼
                            ┌─────────────────────────────┐
                            │  routes_runs.py             │  ✅ REAL
                            │  - save input.mp4           │
                            │  - create RunRecord         │
                            │  - dispatch background task │
                            └─────────────┬───────────────┘
                                          │
                                          ▼
                            ┌─────────────────────────────┐
                            │  orchestrator.run_pipeline()│  ✅ REAL
                            │  serial stage loop          │
                            └─────────────┬───────────────┘
                                          │
        ┌────────────────────┬────────────┼────────────┬────────────────┬────────────┐
        ▼                    ▼            ▼            ▼                ▼            ▼
┌─────────────┐    ┌─────────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌────────┐
│ 1. ingest   │    │ 2. grounding    │ │ 3. track │ │ 4. generation│ │ 5.tempor.│ │ 6.comp │
│             │    │                 │ │          │ │              │ │          │ │        │
│ OpenCV      │    │ Grounding DINO  │ │ SAM 2    │ │ SD-Inpaint   │ │ RAFT     │ │ alpha  │
│ decode      │    │ (transformers)  │ │ video    │ │ + ControlNet │ │ + fwd-bwd│ │ blend  │
│             │    │ first frame +   │ │ predictor│ │ (diffusers)  │ │ warp     │ │        │
│             │    │ full prompt     │ │          │ │              │ │          │ │        │
│             │    │                 │ │          │ │              │ │          │ │        │
│  ✅ REAL    │    │  ✅ REAL        │ │ ✅ REAL  │ │  ✅ REAL     │ │ ✅ REAL  │ │✅ REAL │
│             │    │                 │ │          │ │              │ │  (NEW!)  │ │        │
│ input.mp4   │    │ frame_0 +       │ │ all      │ │ frames +     │ │ frames + │ │frames +│
│   ↓         │    │ prompt          │ │ frames + │ │ masks +      │ │ stylized │ │stable +│
│ frames/     │    │   ↓             │ │ bbox     │ │ prompt       │ │ + masks  │ │ masks  │
│             │    │ bbox            │ │   ↓      │ │   ↓          │ │   ↓      │ │  ↓     │
│             │    │                 │ │ masks/   │ │ stylized/    │ │ stable/  │ │compos./│
└─────────────┘    └─────────────────┘ └──────────┘ └──────────────┘ └──────────┘ └────────┘
                                                                                       │
                                                            ┌──────────┐    ┌──────────┴───┐
                                                            │ 8. eval  │◄───┤ 7. export    │
                                                            │          │    │              │
                                                            │ Farneback│    │ OpenCV       │
                                                            │ + CLIP   │    │ VideoWriter  │
                                                            │ + LPIPS  │    │              │
                                                            │          │    │  ✅ REAL     │
                                                            │ ⚠ partly │    │              │
                                                            │ (mIoU/F  │    │ composited/  │
                                                            │  need GT)│    │     ↓        │
                                                            │          │    │ output/      │
                                                            │ metrics  │    │  final.mp4   │
                                                            │  .json   │    └──────────────┘
                                                            └──────────┘
```

Every stage is an isolated module under `backend/pipeline/`, so each
teammate can iterate on their stage in parallel.

## 3. Repository layout

```
ISY5004-Team18-StylizeIt/
├── backend/                 # FastAPI + processing pipeline
│   ├── main.py                  # uvicorn entry
│   ├── api/                     # REST routes (matches doc/API.md)
│   ├── core/                    # config, schemas, run store
│   ├── pipeline/                # 7 stage modules (ingest → export)
│   ├── evaluation/              # mIoU, F-measure, warping err, CLIP, LPIPS
│   └── utils/
├── frontend/                # Static HTML/Tailwind UI (drag-and-drop demo)
├── configs/                 # YAML presets exposed via /api/v1/configs
├── models/                  # Pretrained SOTA weights (gitignored)
│   ├── grounding_dino/          # Grounding DINO (text → bbox)
│   ├── sam2/                    # SAM 2 (bbox → video mask)
│   ├── stable_diffusion/        # SD-Inpainting / SDXL-Inpainting
│   ├── controlnet/              # Optional structure conditioning
│   ├── raft/                    # Optical flow (temporal stability)
│   ├── clip/                    # CLIP Score (evaluation)
│   └── lpips/                   # Background LPIPS (evaluation)
├── data/                    # Public + self-collected datasets (gitignored)
│   ├── davis2017/               # DAVIS 2017 — tracking benchmark
│   ├── youtube_vos/             # YouTube-VOS — diversity benchmark
│   └── self_collected/          # 10–20 "in-the-wild" phone clips
├── scripts/                 # Helpers: download_models, download_data, run_eval
├── doc/                     # PRD.md, API.md, ARCHITECTURE.md
├── requirements.txt
├── .env.example
└── README.md                # ← you are here
```

## 4. Quick start

### 4.1 Environment

We recommend Python **3.10 or 3.11** in a fresh conda environment.

```powershell
# from project root, Windows PowerShell
conda create -n stylizeit python=3.10 -y
conda activate stylizeit
```

Then run the **all-in-one installer** (also handles the Windows GBK-codec
quirk that breaks the upstream `groundingdino-py` package and pre-fetches
Grounding DINO from Hugging Face):

```powershell
$env:PYTHONUTF8 = "1"
.\scripts\install_packages.ps1
```

Or run the equivalent commands by hand:

```powershell
$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip setuptools wheel
# CUDA 12.1 (drop --index-url for CPU-only)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
# Pre-fetch Grounding DINO via transformers (avoids buggy groundingdino-py)
python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')"
```

> **Why not `pip install groundingdino-py`?** Its `setup.py` reads its
> README with the system codepage and crashes on Windows / CJK locales
> with `UnicodeDecodeError: 'gbk' codec can't decode byte 0xa4`. The exact
> same model ships through `transformers` (`IDEA-Research/grounding-dino-tiny`)
> with no native build step, which is what we use.

### 4.2 Download model weights

One-shot:

```bash
# Linux / macOS / WSL
bash scripts/download_models.sh

# Windows PowerShell
.\scripts\download_models.ps1
```

This fills `models/` with everything the pipeline needs (~7 GB total).
You can re-run it safely — already-downloaded files are skipped.
See [`models/README.md`](models/README.md) for the manual command list.

### 4.3 Download evaluation datasets (optional)

```bash
bash scripts/download_data.sh davis2017      # ~1.7 GB
# YouTube-VOS requires registering at https://youtube-vos.org/ first.
```

For your self-collected clips, drop them into `data/self_collected/clips/`
and update `data/self_collected/manifest.csv`. Privacy rules are listed
in [`data/README.md`](data/README.md).

### 4.4 Run the backend

```bash
# from project root, with the venv active
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

* API docs:  http://localhost:8000/docs
* Frontend (auto-served):  http://localhost:8000/ui
* Artifacts (run outputs):  http://localhost:8000/artifacts/

### 4.5 Run the frontend standalone (optional)

The frontend is a single static HTML file with Tailwind via CDN — no
build step required. Open `frontend/index.html` directly in your browser,
or serve it with any static server. When the backend is reachable at
`http://localhost:8000`, the "Start Run" button submits a real job.

### 4.6 End-to-end smoke test (recommended)

Once you have the backend running:

1. Open http://localhost:8000/ui in your browser.
2. The top-right corner should say "API: online" (green).
3. Click the **Video Drop Zone** and pick any `.mp4` from
   `data/self_collected/clips/` (or any clip <= 10 s / 720p / 30 fps).
4. Type a prompt that matches a visible object, e.g.
   `Turn the person in red into anime style`.
5. Keep `config_name = default.yaml` and hit **Start Run**.
6. Watch the progress bar walk through:
   `ingest → grounding → tracking → generation → temporal → export → evaluation → completed`.
7. The Preview panel auto-plays the result; the Download MP4 button
   appears below it and the Metrics grid fills in.

Everything the run produced is on disk under
`artifacts/runs/<run_id>/` (input.mp4, frames/, masks/, stylized/,
stable/, composited/, output/final.mp4, metrics.json) — great for
debugging individual stages.

**If you don't have a GPU yet:** the pipeline still completes end-to-end
because each stage has a deterministic CPU fallback (rectangle mask /
OpenCV cartoonisation / EMA temporal blend). Metrics and the UI flow are
unaffected, so the frontend can be developed and demoed without a GPU.

## 5. How to use

End-to-end CLI smoke test:

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -F "video=@data/self_collected/clips/clip001.mp4" \
  -F "prompt=Turn the person in red into anime style" \
  -F "config_name=default.yaml" \
  -F "style=anime-only"

# -> {"run_id":"run_a1b2...","status":"queued",...}

curl http://localhost:8000/api/v1/runs/run_a1b2...
# poll every 1-2 s until status == "completed"

curl http://localhost:8000/api/v1/runs/run_a1b2.../result
# -> { output.video_url, metrics: { miou, boundary_f, warping_error, clip_score, background_lpips } }
```

Through the **web UI**: drop a video into the drop-zone, type a prompt,
pick the `default.yaml` preset, hit *Start Run*. The Run Log panel polls
the API every two seconds; once the job completes, the result video and
metric scores appear in the Preview panel.

The full REST contract lives in [`doc/API.md`](doc/API.md).

## 6. SOTA models we depend on

| Stage | Model | Paper | Folder |
| --- | --- | --- | --- |
| Text -> Box | **Grounding DINO** | Liu et al., arXiv:2303.05499 | `models/grounding_dino/` |
| Box -> Video mask | **SAM 2** | Ravi et al., Meta AI 2024 | `models/sam2/` |
| Inpainting | **Stable Diffusion Inpainting / SDXL** | Rombach et al., CVPR 2022 | `models/stable_diffusion/` |
| Structure control | **ControlNet** (canny / openpose) | Zhang et al., ICCV 2023 | `models/controlnet/` |
| Temporal stability | **RAFT** | Teed & Deng, ECCV 2020 | `models/raft/` |
| Eval — prompt alignment | **CLIP ViT-B/32** | Radford et al., ICML 2021 | `models/clip/` |
| Eval — background fidelity | **LPIPS (AlexNet)** | Zhang et al., CVPR 2018 | `models/lpips/` |

## 7. Datasets

| Dataset | Type | Purpose | Source |
| --- | --- | --- | --- |
| **DAVIS 2017** | Public | mIoU, Boundary F-measure for tracking | https://davischallenge.org/davis2017/code.html |
| **YouTube-VOS 2019** | Public | Diversity stress tests | https://youtube-vos.org/dataset/vos/ |
| **Self-collected** | Original | In-the-wild robustness; ~10–20 clips, ~5 s each, phone-recorded | `data/self_collected/` |

## 8. Evaluation metrics

* **mIoU** & **Boundary F-measure** — tracking quality (`scripts/run_eval.py` vs DAVIS GT).
* **Warping Error** — temporal stability (computed automatically for every run).
* **CLIP Score** — prompt-image alignment of the stylized result.
* **Background LPIPS** — perceptual distance on the non-target region (lower = better preservation).

## 9. Development tips

* **No GPU?** The pipeline falls back to a deterministic OpenCV stylizer
  and rectangle masks, so the API, frontend, and metric flow still work
  end-to-end. Great for CI and frontend dev.
* **Where the run lives:** `artifacts/runs/<run_id>/` contains
  `input.mp4`, `frames/`, `masks/`, `stylized/`, `stable/`, `composited/`,
  `output/final.mp4`, and `metrics.json` — everything you need to debug
  a single run.
* **Swap a model** by editing one stage file under `backend/pipeline/`.
  Each stage only reads / writes folders, so they are easy to A/B test.

## 10. License

Code: see [`LICENSE`](LICENSE).
Models and datasets keep their original licenses — listed in
[`models/README.md`](models/README.md) and [`data/README.md`](data/README.md).
This project is for academic use under the ISY5004 module.
