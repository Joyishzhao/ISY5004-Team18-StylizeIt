"""Batch-run self-collected videos through StylizeIt Wan pipeline.

Usage examples (PowerShell):
    python scripts/run_selfcollect_wan_pipeline.py
    python scripts/run_selfcollect_wan_pipeline.py --api-base http://127.0.0.1:8000 --poll-sec 3
    python scripts/run_selfcollect_wan_pipeline.py --output-dir "D:\\StylizeItOutputs\\wan_batch"

What this script does:
1) Scans data/self_collected/clips/*.mp4|*.mov|*.webm
2) POSTs each clip to /api/v1/runs with config_name=wan_vace.yaml
3) Polls run status until completed/failed
4) Downloads each finished output mp4 to a local folder
5) Saves per-run JSON summary (metrics, logs, status)
6) Optionally copies full artifacts/runs/<run_id>/ to local output folder
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}


@dataclass
class JobResult:
    video_path: Path
    run_id: Optional[str]
    status: str
    output_video: Optional[Path]
    summary_json: Path
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_clips = root / "data" / "self_collected" / "clips"
    default_manifest = root / "data" / "self_collected" / "manifest.csv"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = root / "downloads" / f"wan_selfcollect_{stamp}"

    ap = argparse.ArgumentParser(description="Batch-run self-collected videos via Wan VACE.")
    ap.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="StylizeIt backend base URL (default: http://127.0.0.1:8000)",
    )
    ap.add_argument(
        "--input-dir",
        default=str(default_clips),
        help=f"Folder with self-collected clips (default: {default_clips})",
    )
    ap.add_argument(
        "--manifest",
        default=str(default_manifest),
        help="Optional CSV manifest path (for subject/prompt auto-generation).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(default_output),
        help=f"Local output folder to save downloaded results (default: {default_output})",
    )
    ap.add_argument(
        "--config-name",
        default="wan_vace.yaml",
        help="Pipeline config to use. Keep wan_vace.yaml for all-Wan workflow.",
    )
    ap.add_argument(
        "--style",
        default="anime-only",
        help="Style field for API (MVP currently supports anime-only).",
    )
    ap.add_argument(
        "--fallback-prompt",
        default="Turn the main subject into anime style",
        help="Used when prompt cannot be inferred from manifest.",
    )
    ap.add_argument(
        "--poll-sec",
        type=float,
        default=2.0,
        help="Polling interval seconds while waiting each run.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Per-run timeout in seconds.",
    )
    ap.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Also copy artifacts/runs/<run_id> into output-dir/runs/<run_id>.",
    )
    return ap.parse_args()


def _load_manifest_prompts(manifest_path: Path) -> Dict[str, str]:
    if not manifest_path.exists():
        return {}

    prompt_map: Dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            prompt = (row.get("prompt") or "").strip()
            subject = (row.get("subject") or "").strip()
            if prompt:
                prompt_map[filename] = prompt
            elif subject:
                prompt_map[filename] = f"Turn the {subject} into anime style"
    return prompt_map


def _scan_videos(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    videos = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return videos


def _abs_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _create_run(api_base: str, video_path: Path, prompt: str, config_name: str, style: str) -> str:
    url = _abs_url(api_base, "/api/v1/runs")
    with video_path.open("rb") as vf:
        files = {"video": (video_path.name, vf, "video/mp4")}
        data = {
            "prompt": prompt,
            "config_name": config_name,
            "style": style,
        }
        resp = requests.post(url, files=files, data=data, timeout=120)
    resp.raise_for_status()
    body = resp.json()
    run_id = body.get("run_id")
    if not run_id:
        raise RuntimeError(f"API did not return run_id: {body}")
    return str(run_id)


def _poll_until_done(api_base: str, run_id: str, poll_sec: float, timeout_sec: int) -> Dict:
    url = _abs_url(api_base, f"/api/v1/runs/{run_id}")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        info = resp.json()
        status = str(info.get("status", "")).lower()
        stage = info.get("stage", "")
        progress = float(info.get("progress", 0.0))
        print(f"  - run={run_id} status={status} stage={stage} progress={progress:.2f}")
        if status in {"completed", "failed", "cancelled"}:
            return info
        time.sleep(max(0.5, poll_sec))
    raise TimeoutError(f"Run {run_id} timed out after {timeout_sec}s")


def _get_result(api_base: str, run_id: str) -> Dict:
    url = _abs_url(api_base, f"/api/v1/runs/{run_id}/result")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _download_output_video(api_base: str, run_id: str, dest: Path) -> None:
    url = _abs_url(api_base, f"/api/v1/runs/{run_id}/download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def _copy_run_artifacts_if_requested(run_id: str, output_dir: Path, enable: bool) -> Optional[Path]:
    if not enable:
        return None
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "artifacts" / "runs" / run_id
    if not src.exists():
        return None
    dst = output_dir / "runs" / run_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _write_summary(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    api_base = args.api_base.strip().rstrip("/")
    input_dir = Path(args.input_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_videos_dir = output_dir / "videos"
    output_meta_dir = output_dir / "meta"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_videos_dir.mkdir(parents=True, exist_ok=True)
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = _load_manifest_prompts(manifest_path)
    videos = _scan_videos(input_dir)
    if not videos:
        print(f"No videos found in: {input_dir}")
        return 1

    print(f"API base      : {api_base}")
    print(f"Input clips   : {input_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Config        : {args.config_name}")
    print(f"Total clips   : {len(videos)}")
    print("")

    results: List[JobResult] = []
    for idx, video in enumerate(videos, start=1):
        clip_prompt = prompt_map.get(video.name, args.fallback_prompt)
        print(f"[{idx}/{len(videos)}] submit: {video.name}")
        print(f"  - prompt: {clip_prompt}")
        print(f"  - config: {args.config_name}")

        run_id: Optional[str] = None
        status = "failed"
        output_video_path: Optional[Path] = None
        err: Optional[str] = None
        summary_path = output_meta_dir / f"{video.stem}.json"
        summary_payload: Dict = {
            "video": str(video),
            "prompt": clip_prompt,
            "config_name": args.config_name,
            "style": args.style,
            "started_at": datetime.now().isoformat(),
        }

        try:
            run_id = _create_run(
                api_base=api_base,
                video_path=video,
                prompt=clip_prompt,
                config_name=args.config_name,
                style=args.style,
            )
            summary_payload["run_id"] = run_id
            print(f"  - run_id: {run_id}")

            run_info = _poll_until_done(
                api_base=api_base,
                run_id=run_id,
                poll_sec=args.poll_sec,
                timeout_sec=args.timeout_sec,
            )
            status = str(run_info.get("status", "failed"))
            summary_payload["run_info"] = run_info

            if status == "completed":
                result = _get_result(api_base, run_id)
                summary_payload["result"] = result
                output_video_path = output_videos_dir / f"{video.stem}__{run_id}.mp4"
                _download_output_video(api_base, run_id, output_video_path)
                copied = _copy_run_artifacts_if_requested(
                    run_id=run_id,
                    output_dir=output_dir,
                    enable=bool(args.copy_artifacts),
                )
                if copied is not None:
                    summary_payload["copied_artifacts_dir"] = str(copied)
                print(f"  - downloaded: {output_video_path}")
            else:
                err = f"Run ended with status={status}"
                print(f"  - failed: {err}")
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            summary_payload["exception"] = err
            print(f"  - error: {err}")

        summary_payload["final_status"] = status
        summary_payload["finished_at"] = datetime.now().isoformat()
        _write_summary(summary_path, summary_payload)
        results.append(
            JobResult(
                video_path=video,
                run_id=run_id,
                status=status,
                output_video=output_video_path,
                summary_json=summary_path,
                error=err,
            )
        )
        print("")

    done = sum(1 for r in results if r.status == "completed")
    failed = len(results) - done
    index_payload = {
        "api_base": api_base,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config_name": args.config_name,
        "style": args.style,
        "completed": done,
        "failed": failed,
        "total": len(results),
        "items": [
            {
                "video": str(r.video_path),
                "run_id": r.run_id,
                "status": r.status,
                "output_video": str(r.output_video) if r.output_video else None,
                "summary_json": str(r.summary_json),
                "error": r.error,
            }
            for r in results
        ],
    }
    _write_summary(output_dir / "index.json", index_payload)

    print("Batch finished.")
    print(f"  - completed: {done}")
    print(f"  - failed   : {failed}")
    print(f"  - index    : {output_dir / 'index.json'}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
