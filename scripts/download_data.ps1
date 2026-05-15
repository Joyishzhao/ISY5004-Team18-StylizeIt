# Download evaluation datasets into .\data\
# Usage:  .\scripts\download_data.ps1 -Dataset davis2017

param(
  [ValidateSet("davis2017", "youtube_vos")]
  [string]$Dataset = "davis2017"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Data = Join-Path $Root "data"

switch ($Dataset) {
  "davis2017" {
    $Target = Join-Path $Data "davis2017"
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    if (-not (Test-Path (Join-Path $Target "JPEGImages"))) {
      Write-Host "Downloading DAVIS 2017 (480p train/val)..."
      $Zip = Join-Path $Target "DAVIS-2017-trainval-480p.zip"
      Invoke-WebRequest `
        -Uri "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" `
        -OutFile $Zip -UseBasicParsing
      Expand-Archive -Path $Zip -DestinationPath $Target -Force
      Get-ChildItem (Join-Path $Target "DAVIS") | Move-Item -Destination $Target -Force
      Remove-Item (Join-Path $Target "DAVIS") -Recurse -Force
      Remove-Item $Zip
    } else {
      Write-Host "DAVIS 2017 already present, skipping."
    }
  }

  "youtube_vos" {
    Write-Host "YouTube-VOS requires registering at https://youtube-vos.org/dataset/vos/"
    Write-Host "After downloading, place the unzipped 'train\' and 'valid\' folders into:"
    Write-Host "  $Data\youtube_vos\"
  }
}
