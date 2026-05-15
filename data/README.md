# Data Directory

StylizeIt uses **two public video-object-segmentation datasets** for
quantitative evaluation and a **small self-collected pool** for in-the-wild
qualitative testing. Total target volume: ~50 short clips as stated in the
project proposal (Slide 13 of the pitch deck).

## Folder layout

```
data/
├── davis2017/             # Public — downloaded
│   ├── JPEGImages/        # 480p RGB frames per clip
│   ├── Annotations/       # GT segmentation masks
│   └── ImageSets/         # train / val splits
├── youtube_vos/           # Public — downloaded
│   ├── train/
│   ├── valid/
│   │   ├── JPEGImages/
│   │   └── Annotations/
│   └── meta.json
└── self_collected/        # Self-collected — see policy below
    ├── raw/               # original phone recordings (.mov / .mp4)
    ├── clips/             # trimmed <=10s segments (anchor format for inference)
    └── manifest.csv       # filename, scene, subject, duration, fps, has_gt
```

Everything under `data/` is in `.gitignore`. We **only commit the
manifest** and the `README.md`.

## 1) DAVIS 2017 — primary tracking benchmark

* Site: https://davischallenge.org/davis2017/code.html
* Why: high-quality, per-frame object segmentation masks. Used to measure

  official site. Place the unzipped folders into `data/youtube_vos/`.

Recommended subset (~5 GB):
* `valid/` split only -> contains 507 videos with annotations on the first frame.

## 3) Self-collected pool — in-the-wild

* Size target: **10 – 20 clips**, ~5 s each, mixed scenes (campus, street,
  café, transport).
* Capture: any modern smartphone, 1080p or 720p at 30 fps, hand-held OR
  tripod. Save the **original recordings under `data/self_collected/raw/`**.
* Trim each clip to **<=10 s** (matches the MVP API limit) and save the
  trimmed version to `data/self_collected/clips/`.
* Update `manifest.csv` with one row per clip:

```csv
filename,scene,subject,duration_sec,fps,has_gt
clip001.mp4,campus_walk,person_red_jacket,8.5,30,false
clip002.mp4,street_cafe,coffee_cup,5.2,30,false
clip003.mp4,parking_lot,car_blue_sedan,6.0,30,false
```

### Consent / privacy checklist

* No identifiable strangers as the main subject. If a passerby is the
  subject, obtain verbal consent.
* No recording inside private property without permission.
* No minors as the main subject.
* Faces of incidental bystanders should either be (a) far / small, or
  (b) covered later by the stylization output itself.

## Bulk download helper

Both DAVIS and (optionally) the YouTube-VOS validation split are wired
into `scripts/download_data.sh` / `scripts/download_data.ps1`:

```powershell
.\scripts\download_data.ps1 -Dataset davis2017
```

```bash
bash scripts/download_data.sh davis2017
```

The script verifies SHA-256 sums and skips files that already exist so it
is safe to re-run.

## Pointing the backend at a different data root

```bash
export STYLIZEIT_DATA_DIR=/mnt/datasets/stylizeit_data    # Linux / WSL
$env:STYLIZEIT_DATA_DIR = "D:\\stylizeit_data"            # PowerShell
```
