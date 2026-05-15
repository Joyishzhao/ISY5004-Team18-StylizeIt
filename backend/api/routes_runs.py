"""Run lifecycle routes (see doc/API.md §4.3 – §4.7)."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse

from backend.core.config import settings
from backend.core.run_store import run_store
from backend.core.schemas import RunStatus
from backend.pipeline.orchestrator import run_pipeline


router = APIRouter()


ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm"}


def _err(status: int, code: str, message: str, **details) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or {}}},
    )


@router.post("/runs", status_code=201)
async def create_run(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    prompt: str = Form(...),
    config_name: str = Form("default.yaml"),
    style: str = Form("anime-only"),
):
    if not prompt or not prompt.strip():
        return _err(400, "PROMPT_REQUIRED", "Prompt is required.", field="prompt")
    if style != "anime-only":
        return _err(400, "UNSUPPORTED_STYLE", "MVP only supports 'anime-only'.", field="style")
    if video.content_type not in ALLOWED_VIDEO_TYPES:
        return _err(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            f"Unsupported content type: {video.content_type}",
        )

    run_id = run_store.new_run_id()
    run_dir = run_store.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    video_path = run_dir / f"input{suffix}"
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    record = run_store.create(
        run_id=run_id,
        prompt=prompt.strip(),
        config_name=config_name,
        style=style,
        video_path=video_path,
    )

    # Kick off processing in the background. For heavy GPU jobs, swap to
    # Celery / RQ / Ray; the function signature is the same.
    background_tasks.add_task(run_pipeline, run_id)

    return {
        "run_id": record.run_id,
        "status": record.status,
        "stage": record.stage,
        "progress": record.progress,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    record = run_store.get(run_id)
    if record is None:
        return _err(404, "RUN_NOT_FOUND", f"Run not found: {run_id}")
    return {
        "run_id": record.run_id,
        "status": record.status,
        "stage": record.stage,
        "progress": record.progress,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "logs": record.logs,
        "error_message": record.error_message,
    }


@router.get("/runs/{run_id}/result")
def get_run_result(run_id: str):
    record = run_store.get(run_id)
    if record is None:
        return _err(404, "RUN_NOT_FOUND", f"Run not found: {run_id}")
    if record.status != RunStatus.completed:
        return _err(409, "RUN_NOT_COMPLETED", f"Run is {record.status}, not completed.")

    output_video = run_store.run_dir(run_id) / "output" / "final.mp4"
    metrics_path = run_store.run_dir(run_id) / "metrics.json"

    import json
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return {
        "run_id": run_id,
        "status": record.status,
        "output": {
            "video_url": f"/artifacts/runs/{run_id}/output/final.mp4" if output_video.exists() else None,
            "download_url": f"/api/v1/runs/{run_id}/download",
        },
        "metrics": metrics,
    }


@router.get("/runs/{run_id}/download")
def download_run(run_id: str):
    record = run_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="RUN_NOT_FOUND")
    if record.status != RunStatus.completed:
        raise HTTPException(status_code=409, detail="RUN_NOT_COMPLETED")
    output_video = run_store.run_dir(run_id) / "output" / "final.mp4"
    if not output_video.exists():
        raise HTTPException(status_code=404, detail="OUTPUT_NOT_FOUND")
    return FileResponse(
        path=output_video,
        media_type="video/mp4",
        filename=f"{run_id}.mp4",
    )


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    record = run_store.get(run_id)
    if record is None:
        return _err(404, "RUN_NOT_FOUND", f"Run not found: {run_id}")
    if record.status in (RunStatus.completed, RunStatus.failed, RunStatus.cancelled):
        return _err(409, "RUN_NOT_CANCELLABLE", f"Run is {record.status}.")
    updated = run_store.update(run_id, status=RunStatus.cancelled, log="Run cancelled by user.")
    return {
        "run_id": updated.run_id,
        "status": updated.status,
        "stage": updated.stage,
        "updated_at": updated.updated_at.isoformat(),
    }
