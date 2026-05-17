"""Centralised runtime configuration for the StylizeIt backend.

All paths default to locations relative to the project root so the system
works out-of-the-box for local development. Override any field through
environment variables (see `.env.example`) or by editing `configs/*.yaml`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load project-root `.env` so `DASHSCOPE_API_KEY` etc. work without
# `uvicorn --env-file .env` (uvicorn does not load .env by itself).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT

    # Runtime artifacts (uploads, intermediate frames, final mp4)
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("STYLIZEIT_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
    )

    # Static frontend (served at /ui for convenience during dev)
    frontend_dir: Path = field(
        default_factory=lambda: _env_path("STYLIZEIT_FRONTEND_DIR", PROJECT_ROOT / "frontend")
    )

    # Model & dataset roots
    models_dir: Path = field(
        default_factory=lambda: _env_path("STYLIZEIT_MODELS_DIR", PROJECT_ROOT / "models")
    )
    data_dir: Path = field(
        default_factory=lambda: _env_path("STYLIZEIT_DATA_DIR", PROJECT_ROOT / "data")
    )
    configs_dir: Path = field(
        default_factory=lambda: _env_path("STYLIZEIT_CONFIGS_DIR", PROJECT_ROOT / "configs")
    )

    # Inference device: "cuda", "mps", or "cpu"
    device: str = field(default_factory=lambda: os.getenv("STYLIZEIT_DEVICE", "cuda"))

    # Input constraints (kept aligned with doc/API.md)
    max_duration_sec: int = 10
    max_width: int = 1280
    max_height: int = 720
    max_fps: int = 30

    cors_allow_origins: List[str] = field(
        default_factory=lambda: _env_list(
            "STYLIZEIT_CORS_ORIGINS",
            ["http://localhost:5173", "http://localhost:8000", "http://127.0.0.1:8000", "*"],
        )
    )


settings = Settings()
