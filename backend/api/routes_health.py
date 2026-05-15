"""Health check route (see doc/API.md §4.1)."""

from __future__ import annotations

from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True, "service": "stylizeit-api", "version": "v1"}
