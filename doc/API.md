# StylizeIt API Spec (MVP)

This API supports the current frontend workflow:
1. Upload video + prompt + config preset.
2. Start one stylization run.
3. Poll run status.
4. Fetch final output and metrics.

## 1. Base Info
- Base URL: `/api/v1`
- Auth: none (MVP local/dev only)
- Response format: JSON for all non-file endpoints
- Time format: ISO 8601 UTC
- ID format: `run_<12-32 chars>`

## 2. Constraints (MVP)
- Single target object per run.
- Style scope: `anime-only`.
- Input video limits:
  - max duration: `10s`
  - max resolution: `1280x720`
  - max fps: `30`
- Allowed video types: `video/mp4`, `video/quicktime`, `video/webm`

## 3. Common Schemas

### 3.1 Error Response
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Prompt is required.",
    "details": {
      "field": "prompt"
    }
  }
}
```

### 3.2 Run Status Object
```json
{
  "run_id": "run_a1b2c3d4e5f6",
  "status": "queued",
  "stage": "ingest",
  "progress": 0.1,
  "created_at": "2026-03-01T14:00:00Z",
  "updated_at": "2026-03-01T14:00:02Z"
}
```

`status` enum:
- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

`stage` enum:
- `ingest`
- `grounding`
- `tracking`
- `generation`
- `temporal`
- `export`
- `evaluation`
- `completed`

## 4. Endpoints

## 4.1 Health Check
`GET /api/v1/health`

Purpose:
- Frontend/server health probe.

Response `200`:
```json
{
  "ok": true,
  "service": "stylizeit-api",
  "version": "v1"
}
```

## 4.2 List Config Presets
`GET /api/v1/configs`

Purpose:
- Return allowed config presets for frontend selector.

Response `200`:
```json
{
  "configs": [
    {
      "name": "default.yaml",
      "style_scope": "anime-only",
      "limits": {
        "max_duration_sec": 10,
        "max_width": 1280,
        "max_height": 720,
        "max_fps": 30
      }
    }
  ]
}
```

## 4.3 Create Run
`POST /api/v1/runs`

Content-Type:
- `multipart/form-data`

Input fields:
- `video` (file, required)
- `prompt` (string, required)
- `config_name` (string, optional, default `default.yaml`)
- `style` (string, optional, default `anime-only`, must equal `anime-only` in MVP)

Success Response `201`:
```json
{
  "run_id": "run_a1b2c3d4e5f6",
  "status": "queued",
  "stage": "ingest",
  "progress": 0.0,
  "created_at": "2026-03-01T14:00:00Z",
  "updated_at": "2026-03-01T14:00:00Z"
}
```

Error Responses:
- `400 INVALID_INPUT` (missing prompt/file, unsupported style)
- `413 FILE_TOO_LARGE` (video exceeds allowed size policy)
- `415 UNSUPPORTED_MEDIA_TYPE` (unsupported video type)
- `422 VIDEO_LIMIT_EXCEEDED` (duration/resolution/fps above limits)

## 4.4 Get Run Status
`GET /api/v1/runs/{run_id}`

Purpose:
- Poll run lifecycle for status panel.

Success Response `200`:
```json
{
  "run_id": "run_a1b2c3d4e5f6",
  "status": "running",
  "stage": "tracking",
  "progress": 0.48,
  "created_at": "2026-03-01T14:00:00Z",
  "updated_at": "2026-03-01T14:00:09Z",
  "logs": [
    "Ingest completed",
    "Grounding lock acquired",
    "Tracking in progress"
  ]
}
```

Error Response:
- `404 RUN_NOT_FOUND`

## 4.5 Get Run Result
`GET /api/v1/runs/{run_id}/result`

Purpose:
- Fetch output references and evaluation metrics for preview panel.

Success Response `200`:
```json
{
  "run_id": "run_a1b2c3d4e5f6",
  "status": "completed",
  "output": {
    "video_url": "/artifacts/runs/run_a1b2c3d4e5f6/output/final.mp4",
    "download_url": "/api/v1/runs/run_a1b2c3d4e5f6/download"
  },
  "metrics": {
    "miou": 0.62,
    "boundary_f": 0.68,
    "warping_error": 0.14,
    "clip_score": 0.29,
    "background_lpips": 0.10
  }
}
```

Error Responses:
- `404 RUN_NOT_FOUND`
- `409 RUN_NOT_COMPLETED`

## 4.6 Download Final Video
`GET /api/v1/runs/{run_id}/download`

Purpose:
- Download final stylized video file.

Success Response:
- `200` with file stream (`video/mp4`)

Error Responses:
- `404 RUN_NOT_FOUND`
- `409 RUN_NOT_COMPLETED`

## 4.7 Cancel Run
`POST /api/v1/runs/{run_id}/cancel`

Purpose:
- Cancel queued/running run.

Success Response `200`:
```json
{
  "run_id": "run_a1b2c3d4e5f6",
  "status": "cancelled",
  "stage": "completed",
  "updated_at": "2026-03-01T14:00:15Z"
}
```

Error Responses:
- `404 RUN_NOT_FOUND`
- `409 RUN_NOT_CANCELLABLE`

## 5. Frontend Integration Contract
- On click `Start Run`:
  1. `POST /api/v1/runs`
  2. Poll `GET /api/v1/runs/{run_id}` every 1-2 seconds.
  3. On `status=completed`, call `GET /api/v1/runs/{run_id}/result`.
  4. Show `output.video_url` in preview and metrics in cards.
  5. On `failed`, render `error.message`.

## 6. Failure Code Catalog
- `INVALID_INPUT`
- `PROMPT_REQUIRED`
- `VIDEO_REQUIRED`
- `UNSUPPORTED_STYLE`
- `UNSUPPORTED_MEDIA_TYPE`
- `VIDEO_LIMIT_EXCEEDED`
- `RUN_NOT_FOUND`
- `RUN_NOT_COMPLETED`
- `RUN_NOT_CANCELLABLE`
- `INTERNAL_ERROR`

## 7. Example Create-Run Request (cURL)
```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -F "video=@sample.mp4" \
  -F "prompt=Turn the person in red into anime style" \
  -F "config_name=default.yaml" \
  -F "style=anime-only"
```
