"""Pydantic schemas shared across API routes and the pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class RunStage(str, Enum):
    ingest = "ingest"
    grounding = "grounding"
    tracking = "tracking"
    generation = "generation"
    temporal = "temporal"
    export = "export"
    evaluation = "evaluation"
    completed = "completed"


class RunRecord(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.queued
    stage: RunStage = RunStage.ingest
    progress: float = 0.0
    created_at: datetime
    updated_at: datetime
    prompt: str
    config_name: str = "default.yaml"
    style: str = "anime-only"
    video_path: str
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class Metrics(BaseModel):
    miou: Optional[float] = None
    boundary_f: Optional[float] = None
    warping_error: Optional[float] = None
    clip_score: Optional[float] = None
    background_lpips: Optional[float] = None


class RunResult(BaseModel):
    run_id: str
    status: RunStatus
    output_video_url: Optional[str] = None
    download_url: Optional[str] = None
    metrics: Metrics = Field(default_factory=Metrics)
