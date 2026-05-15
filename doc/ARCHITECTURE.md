# StylizeIt — Architecture & Workflow

This document is the **engineering** companion to `PRD.md` (product) and
`API.md` (contract). It describes how the codebase is wired together so
team members can pick a stage and iterate without stepping on each other.

## 1. Top-level view

```
┌────────────────────────┐        ┌───────────────────────────────────────────┐
│  Frontend (static HTML)│  HTTP  │ Backend (FastAPI, backend/main.py)        │
│  frontend/index.html   │ ─────▶ │  /api/v1/runs   (POST start)              │
│  Tailwind + JS poll    │ ◀───── │  /api/v1/runs/{id}  (GET status)          │
│                        │        │  /api/v1/runs/{id}/result (GET artifacts) │
└────────────────────────┘        │  /artifacts/...  (static files)           │
                                  └─────────────┬─────────────────────────────┘
                                                │ background task
                                                ▼
                              ┌────────────────────────────────────────┐
                              │ backend/pipeline/orchestrator.py       │
                              │  (sequential stage runner)             │
                              └────────────────────────────────────────┘
                                                │
        ┌───────────┬─────────────┬─────────────┼──────────────┬──────────────┬─────────────┐
        ▼           ▼             ▼             ▼              ▼              ▼             ▼
   ingest.py  grounding.py   tracking.py   generation.py   temporal.py   compositor.py  export.py
                                                                                          │
                                                                                          ▼
                                                                              evaluation/metrics.py
```

## 2. Request lifecycle

1. **Upload** — frontend `POST /api/v1/runs` with `multipart/form-data` (video + prompt).
2. **Validation** — `routes_runs.py` checks MIME / style / prompt.
3. **Persist** — video saved to `artifacts/runs/<run_id>/input.mp4`;
   metadata into the in-memory `RunStore` (`backend/core/run_store.py`).
4. **Dispatch** — FastAPI `BackgroundTasks` calls
   `backend.pipeline.orchestrator.run_pipeline(run_id)`.
5. **Stages** — each stage reads / writes a sub-folder under the run dir
   and updates progress (`stage`, `progress`, `logs`) via `run_store.update`.
6. **Polling** — frontend hits `GET /api/v1/runs/{run_id}` every 1-2 s.
7. **Result** — when status flips to `completed`, frontend calls
   `GET /api/v1/runs/{run_id}/result` and renders the video + metrics.

## 3. Folder contract per run

```
artifacts/runs/<run_id>/
├── input.mp4              # original upload
├── frames/                # 00000.png ... NNNNN.png (RGB)
├── masks/                 # 00000.png ... NNNNN.png (binary, white = target)
├── stylized/              # per-frame diffusion outputs
├── stable/                # temporally-smoothed stylized frames
├── composited/            # stylized target + original background
├── output/final.mp4       # served as the result
└── metrics.json           # mIoU / F / warping_err / clip / lpips
```

Each stage **only** reads the folders before it. This makes them trivially
swappable (e.g. replace `tracking.py` with an XMem variant — no other
file needs to change).

## 4. Stage responsibilities

| Stage file | Reads | Writes | SOTA model |
| --- | --- | --- | --- |
| `ingest.py` | `input.mp4` | `frames/`, `ctx.fps/width/height` | — (OpenCV) |
| `grounding.py` | `frames/00000.png`, `prompt` | `ctx.initial_bbox` | Grounding DINO |
| `tracking.py` | `frames/`, `ctx.initial_bbox` | `masks/` | SAM 2 (video) |
| `generation.py` | `frames/`, `masks/`, `prompt` | `stylized/` | SD/SDXL Inpaint + ControlNet (optional) |
| `temporal.py` | `frames/`, `stylized/`, `masks/` | `stable/` (overrides `ctx.stylized_dir`) | RAFT |
| `compositor.py` | `frames/`, `stylized/`, `masks/` | `composited/` | — (alpha blend) |
| `export.py` | `composited/` | `output/final.mp4` | — (OpenCV writer) |
| `evaluation/metrics.py` | `frames/`, `stylized/`, `masks/` | `ctx.metrics` | RAFT, CLIP, LPIPS |

## 5. Configuration

* `backend/core/config.py` defines `Settings` (paths, device, limits).
  Override via env vars (see `.env.example`) — never hard-code.
* `configs/*.yaml` are exposed via `GET /api/v1/configs` so the frontend
  selector is always in sync with the backend.

## 6. Failure & fallback policy

* Every stage that wraps a SOTA model first checks for its weights under
  `models/<name>/`. If they are missing, the stage logs a warning and
  produces a **deterministic placeholder** output (rectangle mask /
  OpenCV stylization / EMA blend). This keeps the API contract intact
  even when running on a laptop with no GPU.
* Any unhandled exception inside `run_pipeline` is caught by the
  orchestrator, written to `run.error_message`, and surfaced via the
  status endpoint — the API never 500s on the user.

## 7. Scaling beyond MVP (out of scope, but designed for)

| Concern | MVP | Future |
| --- | --- | --- |
| Job queue | `BackgroundTasks` | Celery / RQ / Ray Serve |
| Run store | In-process dict | SQLite / Postgres + Redis |
| GPU pool | Single worker | Per-stage workers (SAM 2 box, SD box) |
| Storage | Local disk | S3-compatible (artifact URL stays the same) |

Because every stage already communicates through files, switching to a
distributed setup mostly means swapping `BackgroundTasks` for a real
queue and the `RunStore` for a DB — no pipeline code changes.
