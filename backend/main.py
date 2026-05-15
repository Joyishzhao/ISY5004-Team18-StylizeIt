"""FastAPI entry point for the StylizeIt backend.

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import routes_configs, routes_health, routes_runs
from backend.core.config import settings
from backend.utils.logging import configure_root_logging


configure_root_logging()
log = logging.getLogger("stylizeit")


def create_app() -> FastAPI:
    app = FastAPI(
        title="StylizeIt API",
        version="0.1.0",
        description=(
            "Backend API for StylizeIt: target-level video stylization. "
            "Implements the workflow defined in doc/API.md."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = "/api/v1"
    app.include_router(routes_health.router, prefix=api_prefix, tags=["health"])
    app.include_router(routes_configs.router, prefix=api_prefix, tags=["configs"])
    app.include_router(routes_runs.router, prefix=api_prefix, tags=["runs"])

    # Serve produced artifacts (final mp4, intermediate frames) directly.
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/artifacts",
        StaticFiles(directory=str(settings.artifacts_dir)),
        name="artifacts",
    )

    # Optional: serve the static frontend for one-click local demo.
    if settings.frontend_dir.exists():
        app.mount(
            "/ui",
            StaticFiles(directory=str(settings.frontend_dir), html=True),
            name="ui",
        )

    @app.on_event("startup")
    async def _print_banner() -> None:
        log.info("StylizeIt backend ready.")
        log.info("  API docs : http://localhost:8000/docs")
        log.info("  Frontend : http://localhost:8000/ui")
        log.info("  Artifacts: %s", settings.artifacts_dir)
        log.info("  Models   : %s", settings.models_dir)
        log.info("  Device   : %s", settings.device)

    return app


app = create_app()
