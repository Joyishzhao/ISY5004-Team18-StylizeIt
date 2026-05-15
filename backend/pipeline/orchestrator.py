"""End-to-end pipeline orchestrator.

This file glues together every stage. Each stage is intentionally a thin
wrapper around a SOTA model so the team can iterate on them independently:

    ingest -> grounding -> tracking -> generation -> temporal ->
    compositor -> export -> evaluation

Stages communicate through a `PipelineContext` dataclass (just paths and
numpy arrays) so they remain decoupled.

NOTE: The default implementations are reference scaffolds. They run end-to-end
on CPU but produce a placeholder output. Replace the body of each stage
function under `backend/pipeline/<stage>.py` with the real SOTA model call.
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from backend.core.run_store import run_store
from backend.core.schemas import RunStage, RunStatus
from backend.pipeline import (
    compositor,
    export,
    generation,
    grounding,
    ingest,
    temporal,
    tracking,
)
from backend.evaluation import metrics as eval_metrics


log = logging.getLogger("stylizeit.pipeline")


@dataclass
class PipelineContext:
    run_id: str
    run_dir: Path
    video_path: Path
    prompt: str
    style: str
    config_name: str

    # Filled in stage-by-stage
    frames_dir: Path = None  # type: ignore[assignment]
    fps: float = 0.0
    width: int = 0
    height: int = 0

    initial_bbox: List[float] = field(default_factory=list)
    masks_dir: Path = None  # type: ignore[assignment]
    stylized_dir: Path = None  # type: ignore[assignment]
    composited_dir: Path = None  # type: ignore[assignment]
    output_path: Path = None  # type: ignore[assignment]

    metrics: Dict[str, Any] = field(default_factory=dict)


def _update(run_id: str, stage: RunStage, progress: float, log_msg: str) -> None:
    run_store.update(run_id, stage=stage, progress=progress, log=log_msg, status=RunStatus.running)


def run_pipeline(run_id: str) -> None:
    """Background task entry point.

    Errors are caught and surfaced through the run store so the API can
    report them via `GET /runs/{run_id}`.
    """

    record = run_store.get(run_id)
    if record is None:
        log.error("Run not found in store: %s", run_id)
        return

    ctx = PipelineContext(
        run_id=run_id,
        run_dir=run_store.run_dir(run_id),
        video_path=Path(record.video_path),
        prompt=record.prompt,
        style=record.style,
        config_name=record.config_name,
    )

    try:
        _update(run_id, RunStage.ingest, 0.05, "Ingest started")
        ingest.run(ctx)

        _update(run_id, RunStage.grounding, 0.20, "Grounding (text -> box)")
        grounding.run(ctx)

        _update(run_id, RunStage.tracking, 0.40, "Tracking + segmentation")
        tracking.run(ctx)

        _update(run_id, RunStage.generation, 0.65, "Diffusion stylization")
        generation.run(ctx)

        _update(run_id, RunStage.temporal, 0.80, "Temporal stabilization")
        temporal.run(ctx)

        _update(run_id, RunStage.export, 0.92, "Compositing + export")
        compositor.run(ctx)
        export.run(ctx)

        _update(run_id, RunStage.evaluation, 0.97, "Evaluating metrics")
        eval_metrics.run(ctx)

        (ctx.run_dir / "metrics.json").write_text(
            json.dumps(ctx.metrics, indent=2), encoding="utf-8"
        )

        run_store.update(
            run_id,
            status=RunStatus.completed,
            stage=RunStage.completed,
            progress=1.0,
            log="Run completed successfully.",
        )
    except Exception as exc:  # noqa: BLE001 - we want to surface everything
        tb = traceback.format_exc()
        log.exception("Pipeline failed for run %s", run_id)
        run_store.update(
            run_id,
            status=RunStatus.failed,
            log=f"Pipeline failed: {exc}",
            error_message=tb,
        )
