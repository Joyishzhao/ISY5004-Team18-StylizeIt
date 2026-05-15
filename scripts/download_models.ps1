# Download all model weights into .\models\
# Usage:   .\scripts\download_models.ps1
# Re-running is safe -- existing files are skipped.

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Models = Join-Path $Root "models"

$dirs = @("grounding_dino", "sam2", "stable_diffusion", "controlnet", "raft", "clip", "lpips")
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $Models $d) | Out-Null }

function Download-IfMissing($Url, $Dest) {
  if (-not (Test-Path $Dest)) {
    Write-Host "  -> $Dest"
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
  } else {
    Write-Host "  (exists) $Dest"
  }
}

Write-Host "[1/5] Grounding DINO"
Download-IfMissing `
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" `
  (Join-Path $Models "grounding_dino\groundingdino_swint_ogc.pth")
Download-IfMissing `
  "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py" `
  (Join-Path $Models "grounding_dino\GroundingDINO_SwinT_OGC.py")

Write-Host "[2/5] SAM 2 (hiera_large)"
Download-IfMissing `
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" `
  (Join-Path $Models "sam2\sam2.1_hiera_large.pt")

Write-Host "[3/5] Stable Diffusion Inpainting"
python -c @"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='runwayml/stable-diffusion-inpainting',
                  local_dir=r'$($Models)\stable_diffusion\sd-inpainting',
                  local_dir_use_symlinks=False)
"@

Write-Host "[4/5] ControlNet (canny)"
python -c @"
from huggingface_hub import snapshot_download
snapshot_download(repo_id='lllyasviel/control_v11p_sd15_canny',
                  local_dir=r'$($Models)\controlnet\canny',
                  local_dir_use_symlinks=False)
"@

Write-Host "[5/5] RAFT (things)"
try {
  Download-IfMissing `
    "https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-things.pth" `
    (Join-Path $Models "raft\raft-things.pth")
} catch {
  Write-Host "  (skipped: RAFT release missing -- torchvision RAFT will be used at runtime)"
}

Write-Host ""
Write-Host "Done. Model tree:"
Get-ChildItem -Path $Models -Recurse -File | Select-Object FullName, Length | Format-Table -AutoSize
