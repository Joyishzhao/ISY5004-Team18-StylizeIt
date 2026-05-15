#!/usr/bin/env bash
# Download evaluation datasets into ./data/
# Usage:   bash scripts/download_data.sh [davis2017|youtube_vos]
# Default: davis2017

set -euo pipefail
DATASET="${1:-davis2017}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"

case "$DATASET" in
  davis2017)
    mkdir -p "$DATA/davis2017"
    cd "$DATA/davis2017"
    if [ ! -d JPEGImages ]; then
      echo "Downloading DAVIS 2017 (480p train/val)..."
      curl -L --fail -o DAVIS-2017-trainval-480p.zip \
        https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
      unzip -q DAVIS-2017-trainval-480p.zip
      mv DAVIS/* . || true
      rm -rf DAVIS DAVIS-2017-trainval-480p.zip
    else
      echo "DAVIS 2017 already present, skipping."
    fi
    ;;

  youtube_vos)
    echo "YouTube-VOS requires registering at https://youtube-vos.org/dataset/vos/"
    echo "After downloading, place the unzipped 'train/' and 'valid/' folders into:"
    echo "  $DATA/youtube_vos/"
    ;;

  *)
    echo "Unknown dataset: $DATASET"
    echo "Usage: bash scripts/download_data.sh [davis2017|youtube_vos]"
    exit 1
    ;;
esac
