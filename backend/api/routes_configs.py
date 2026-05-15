"""Config preset listing route (see doc/API.md §4.2)."""

from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import settings


router = APIRouter()


@router.get("/configs")
def list_configs() -> dict:
    presets = []
    if settings.configs_dir.exists():
        for path in sorted(settings.configs_dir.glob("*.yaml")):
            presets.append(
                {
                    "name": path.name,
                    "style_scope": "anime-only",
                    "limits": {
                        "max_duration_sec": settings.max_duration_sec,
                        "max_width": settings.max_width,
                        "max_height": settings.max_height,
                        "max_fps": settings.max_fps,
                    },
                }
            )
    if not presets:
        presets.append(
            {
                "name": "default.yaml",
                "style_scope": "anime-only",
                "limits": {
                    "max_duration_sec": settings.max_duration_sec,
                    "max_width": settings.max_width,
                    "max_height": settings.max_height,
                    "max_fps": settings.max_fps,
                },
            }
        )
    return {"configs": presets}
