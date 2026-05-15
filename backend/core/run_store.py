"""In-memory store for run metadata.

This is intentionally lightweight so the MVP can run without a database.
Each run also has a folder on disk under `<artifacts_dir>/runs/<run_id>/`.

Swap this out for SQLite / Redis when scaling beyond a single worker.
"""

from __future__ import annotations

import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from backend.core.config import settings
from backend.core.schemas import RunRecord, RunStage, RunStatus


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class RunStore:
    def __init__(self) -> None:
        self._records: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_run_id() -> str:
        return "run_" + secrets.token_hex(8)

    def run_dir(self, run_id: str) -> Path:
        return settings.artifacts_dir / "runs" / run_id

    def create(
        self,
        run_id: str,
        prompt: str,
        config_name: str,
        style: str,
        video_path: Path,
    ) -> RunRecord:
        record = RunRecord(
            run_id=run_id,
            status=RunStatus.queued,
            stage=RunStage.ingest,
            progress=0.0,
            created_at=_now(),
            updated_at=_now(),
            prompt=prompt,
            config_name=config_name,
            style=style,
            video_path=str(video_path),
        )
        with self._lock:
            self._records[run_id] = record
        return record

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._records.get(run_id)

    def update(
        self,
        run_id: str,
        *,
        status: Optional[RunStatus] = None,
        stage: Optional[RunStage] = None,
        progress: Optional[float] = None,
        log: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[RunRecord]:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return None
            if status is not None:
                record.status = status
            if stage is not None:
                record.stage = stage
            if progress is not None:
                record.progress = max(0.0, min(1.0, progress))
            if log is not None:
                record.logs.append(log)
            if error_message is not None:
                record.error_message = error_message
            record.updated_at = _now()
            return record


run_store = RunStore()
