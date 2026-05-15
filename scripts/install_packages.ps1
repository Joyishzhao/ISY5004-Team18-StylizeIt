# StylizeIt -- one-shot package installer for Windows (PowerShell).
# Run from project root inside an activated conda/venv environment:
#   conda activate stylizeit
#   .\scripts\install_packages.ps1
#
# This script handles the Windows GBK-codec quirk that breaks `pip install groundingdino-py`
# by using the transformers implementation of Grounding DINO instead.

$ErrorActionPreference = "Stop"

Write-Host "==> Step 1: upgrade pip and force UTF-8 for sub-builds" -ForegroundColor Cyan
$env:PYTHONUTF8 = "1"
python -m pip install --upgrade pip setuptools wheel

Write-Host ""
Write-Host "==> Step 2: install PyTorch (CUDA 12.1 build)" -ForegroundColor Cyan
Write-Host "    If you don't have a CUDA GPU, replace --index-url with the CPU wheel."
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "==> Step 3: install everything in requirements.txt" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "==> Step 4: install SAM 2 from GitHub (not on PyPI)" -ForegroundColor Cyan
pip install "git+https://github.com/facebookresearch/sam2.git"

Write-Host ""
Write-Host "==> Step 5: pre-fetch the Grounding DINO HF snapshot (~700 MB)" -ForegroundColor Cyan
python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); print('Grounding DINO ready')"

Write-Host ""
Write-Host "==> Step 6: verify install" -ForegroundColor Cyan
python -c "import torch, fastapi, diffusers, transformers, cv2, lpips; print('torch', torch.__version__, 'CUDA?', torch.cuda.is_available()); print('fastapi', fastapi.__version__); print('diffusers', diffusers.__version__); print('transformers', transformers.__version__); print('opencv', cv2.__version__); print('lpips OK')"

try {
    python -c "import sam2; print('sam2 OK')"
} catch {
    Write-Host "WARNING: SAM 2 import failed -- tracking will fall back to rectangle mask." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "All set. Next: .\scripts\download_models.ps1 (if not already done)." -ForegroundColor Green
