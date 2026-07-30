<#
.SYNOPSIS
    Descarga un segundo dataset de melanoma (Kaggle) para diversificar las
    imágenes de ejemplo de la demo, sin usarlo para reentrenar el modelo.

.DESCRIPTION
    Dataset: "Melanoma Detection Dataset" (wanderdust), basado en el reto
    ISIC 2017 (Codella et al. 2018). Licencia CC0: Public Domain.
    https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection

    Estructura: skin-lesions/{train,valid,test}/{melanoma,nevus,seborrheic_keratosis}/
    3 clases, no 2: para las etiquetas benigno/maligno de la demo,
    scripts/refresh-samples.mjs mapea melanoma -> maligno y
    nevus + seborrheic_keratosis -> benigno (ambas son lesiones benignas).

    Mismos requisitos que download_dataset.ps1 (cuenta Kaggle + kaggle.json
    en %USERPROFILE%\.kaggle\kaggle.json).

.NOTES
    AVISO DE TAMAÑO: el dataset completo pesa ~12.2 GB (~2750 imágenes en
    train+valid+test). La API de Kaggle no permite descargar una carpeta
    suelta, así que se descarga entero aunque solo se necesiten unas pocas
    docenas de imágenes de ejemplo. Tarda y ocupa disco; no se versiona
    (excluido en .gitignore, igual que dataset/).
#>

$ErrorActionPreference = 'Stop'

$repoRoot   = Split-Path -Parent $PSScriptRoot
$datasetDir = Join-Path $repoRoot 'dataset2'
$kaggleJson = Join-Path $env:USERPROFILE '.kaggle\kaggle.json'

if (-not (Test-Path $kaggleJson)) {
    Write-Host "`nFALTA kaggle.json" -ForegroundColor Red
    Write-Host "  1) Ve a https://www.kaggle.com/settings/account"
    Write-Host "  2) Pulsa 'Create New Token' en la seccion API"
    Write-Host "  3) Mueve el kaggle.json descargado a: $kaggleJson"
    Write-Host "  4) Vuelve a ejecutar este script.`n"
    exit 1
}

$pyOk = $false
foreach ($cmd in @('python', 'py')) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found -and (& $cmd --version 2>$null)) { $pyOk = $true; $py = $cmd; break }
}
if (-not $pyOk) {
    Write-Host "Python no esta instalado. Instalalo desde https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando Kaggle CLI..." -ForegroundColor Cyan
    & $py -m pip install --user kaggle
}

New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null
Write-Host "Descargando dataset (~12.2 GB, puede tardar bastante)..." -ForegroundColor Yellow
kaggle datasets download `
    -d wanderdust/skin-lesion-analysis-toward-melanoma-detection `
    -p $datasetDir `
    --unzip

Write-Host "`nListo. Dataset en: $datasetDir" -ForegroundColor Green
Write-Host "Ahora ejecuta: node scripts/refresh-samples.mjs`n"
Get-ChildItem $datasetDir | Format-Table -AutoSize
