"""Logging helpers (kept minimal; uvicorn already wires up a sensible setup)."""

from __future__ import annotations

import logging


def configure_root_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
