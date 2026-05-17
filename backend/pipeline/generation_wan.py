"""Stage 4 (Wan VACE) — video-native stylization via Alibaba DashScope.

Uses wan2.1-vace-plus with function ``video_edit`` (local editing):
  - input: source video URL + mask image URL (frame 0, white = edit region)
  - mask_type=tracking: Wan tracks the target internally (built-in temporal model)
  - output: temporally coherent MP4 (no per-frame SD + no RAFT needed)

Docs:
  https://www.alibabacloud.com/help/en/model-studio/wan-vace-guide

Requirements:
  - DASHSCOPE_API_KEY in environment (Singapore: wan2.1-vace-plus)
  - STYLIZEIT_PUBLIC_BASE_URL: publicly reachable base URL for this API server
    (e.g. ngrok https://xxxx.ngrok-free.app) so DashScope can fetch video/mask files
    under /artifacts/runs/<run_id>/...
  - Input video <= 5 seconds (Wan limit); longer clips are auto-trimmed.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import requests

from backend.core.config import settings

log = logging.getLogger("stylizeit.generation_wan")

# Region endpoints (model, key, and URL must match — see Alibaba docs).
_ENDPOINTS = {
    "intl": {
        "base_url": "https://dashscope-intl.aliyuncs.com/api/v1",
        "model": "wan2.1-vace-plus",
    },
    "cn": {
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "model": "wanx2.1-vace-plus",
    },
}


def _load_yaml_config(config_name: str) -> dict:
    try:
        import yaml  # type: ignore

        path = settings.configs_dir / config_name
        if not path.exists():
            path = settings.configs_dir / "wan_vace.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read config: %s", exc)
    return {}


def _public_url(rel_path: str) -> str:
    """Build a URL DashScope can download (must NOT be localhost-only)."""
    base = os.getenv("STYLIZEIT_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "[wan_vace] STYLIZEIT_PUBLIC_BASE_URL is not set. "
            "DashScope must fetch your video and mask over HTTPS. "
            "Expose this server (e.g. ngrok http 8000) and set "
            "STYLIZEIT_PUBLIC_BASE_URL=https://<your-tunnel-host> in .env"
        )
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"{base}/{rel}"


def _trim_video_for_wan(ctx, max_sec: float) -> Path:
    """Write wan_input.mp4 trimmed to Wan duration limit."""
    out = ctx.run_dir / "wan_input.mp4"
    cap = cv2.VideoCapture(str(ctx.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[wan_vace] cannot open {ctx.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or ctx.fps or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or ctx.width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or ctx.height
    max_frames = int(max_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()

    if n == 0:
        raise RuntimeError("[wan_vace] trimmed video has 0 frames.")
    log.info("[wan_vace] trimmed input: %d frames (%.2fs @ %.1f fps) -> %s", n, n / fps, fps, out)
    return out


def _export_mask_frame0(ctx) -> Path:
    """Pick first-frame mask for Wan (white = edit)."""
    masks = sorted(ctx.masks_dir.glob("*.png"))
    if not masks:
        raise RuntimeError("[wan_vace] no masks/ — run tracking first.")
    src = masks[0]
    dst = ctx.run_dir / "wan_mask_frame0.png"
    shutil.copy2(src, dst)
    return dst


def _create_task(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    video_url: str,
    mask_url: str,
    mask_frame_id: int,
    mask_type: str,
    expand_ratio: float,
    prompt_extend: bool,
) -> str:
    payload = {
        "model": model,
        "input": {
            "function": "video_edit",
            "prompt": prompt,
            "video_url": video_url,
            "mask_image_url": mask_url,
            "mask_frame_id": mask_frame_id,
        },
        "parameters": {
            "prompt_extend": prompt_extend,
            "mask_type": mask_type,
            "expand_ratio": expand_ratio,
        },
    }
    resp = requests.post(
        f"{base_url}/services/aigc/video-generation/video-synthesis",
        headers={
            "X-DashScope-Async": "enable",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "output" not in data or "task_id" not in data["output"]:
        raise RuntimeError(f"[wan_vace] unexpected create response: {data}")
    return data["output"]["task_id"]


def _poll_task(
    *,
    base_url: str,
    api_key: str,
    task_id: str,
    poll_interval: int = 15,
    timeout_sec: int = 600,
) -> str:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = requests.get(
            f"{base_url}/tasks/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        out = resp.json().get("output", {})
        status = out.get("task_status", "UNKNOWN")
        log.info("[wan_vace] task %s status=%s", task_id, status)
        if status == "SUCCEEDED":
            url = out.get("video_url")
            if not url:
                raise RuntimeError(f"[wan_vace] SUCCEEDED but no video_url: {out}")
            return url
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"[wan_vace] task {status}: {out.get('message', out)}")
        time.sleep(poll_interval)
    raise RuntimeError(f"[wan_vace] task {task_id} timed out after {timeout_sec}s")


def _download_video(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    log.info("[wan_vace] downloaded result -> %s", dest)


def _extract_frames_for_eval(mp4: Path, stylized_dir: Path, fps: float) -> None:
    """Populate stylized/ so evaluation metrics still work."""
    stylized_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(stylized_dir / f"{idx:05d}.png"), frame)
        idx += 1
    cap.release()
    log.info("[wan_vace] extracted %d frames to %s for evaluation.", idx, stylized_dir)


def run(ctx) -> None:  # type: ignore[no-untyped-def]
    cfg = _load_yaml_config(getattr(ctx, "config_name", "wan_vace.yaml"))
    wan_cfg = cfg.get("generation", {}).get("wan_vace", {}) or cfg.get("wan_vace", {}) or {}

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "[wan_vace] DASHSCOPE_API_KEY is not set. "
            "Get a key from Alibaba Model Studio and add it to .env"
        )

    region = wan_cfg.get("region", "intl")
    ep = _ENDPOINTS.get(region, _ENDPOINTS["intl"])
    max_sec = float(wan_cfg.get("max_duration_sec", 5.0))
    # Wan API uses 1-based frame index (1 = first frame). Our pipeline frames are 0-based.
    raw_mask_frame = int(wan_cfg.get("mask_frame_id", 1))
    mask_frame_id = raw_mask_frame if raw_mask_frame >= 1 else raw_mask_frame + 1
    mask_type = str(wan_cfg.get("mask_type", "tracking"))
    expand_ratio = float(wan_cfg.get("expand_ratio", 0.05))
    prompt_extend = bool(wan_cfg.get("prompt_extend", False))
    poll_interval = int(wan_cfg.get("poll_interval_sec", 15))
    poll_timeout = int(wan_cfg.get("poll_timeout_sec", 600))

    # User prompt carries the creative intent; suffix is optional quality hints only.
    style_suffix = str(wan_cfg.get("style_prompt_suffix", "") or "").strip()
    user_prompt = ctx.prompt.strip()
    if style_suffix:
        full_prompt = f"{user_prompt}. {style_suffix}"
    else:
        full_prompt = user_prompt
    log.info("[wan_vace] prompt sent to API: %s", full_prompt[:200])

    # 1) Prepare assets on disk
    wan_video = _trim_video_for_wan(ctx, max_sec)
    wan_mask = _export_mask_frame0(ctx)

    # 2) Public URLs (artifacts are mounted at /artifacts/ by FastAPI)
    run_rel = f"artifacts/runs/{ctx.run_id}"
    video_url = _public_url(f"{run_rel}/wan_input.mp4")
    mask_url = _public_url(f"{run_rel}/wan_mask_frame0.png")
    log.info("[wan_vace] video_url=%s", video_url)
    log.info("[wan_vace] mask_url=%s", mask_url)

    # 3) Async API
    task_id = _create_task(
        base_url=ep["base_url"],
        api_key=api_key,
        model=ep["model"],
        prompt=full_prompt,
        video_url=video_url,
        mask_url=mask_url,
        mask_frame_id=mask_frame_id,
        mask_type=mask_type,
        expand_ratio=expand_ratio,
        prompt_extend=prompt_extend,
    )
    log.info("[wan_vace] created task_id=%s", task_id)
    (ctx.run_dir / "wan_task_id.txt").write_text(task_id, encoding="utf-8")

    result_url = _poll_task(
        base_url=ep["base_url"],
        api_key=api_key,
        task_id=task_id,
        poll_interval=poll_interval,
        timeout_sec=poll_timeout,
    )

    # 4) Save final output
    output_dir = ctx.run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = output_dir / "final.mp4"
    _download_video(result_url, final_mp4)
    ctx.output_path = final_mp4

    # 5) Frames for UI preview + metrics (optional but useful)
    stylized_dir = ctx.run_dir / "stylized"
    ctx.stylized_dir = stylized_dir
    _extract_frames_for_eval(final_mp4, stylized_dir, ctx.fps or 30.0)

    # Wan already composites edited region onto background — treat as composited too.
    composited_dir = ctx.run_dir / "composited"
    composited_dir.mkdir(parents=True, exist_ok=True)
    for fp in stylized_dir.glob("*.png"):
        shutil.copy2(fp, composited_dir / fp.name)
    ctx.composited_dir = composited_dir

    log.info("[wan_vace] done. output=%s (temporal/RAFT skipped — video model).", final_mp4)
