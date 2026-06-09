<#
.SYNOPSIS
    Descarga el dataset de melanomas desde Kaggle.

.DESCRIPTION
    Auto-instala el CLI de Kaggle si hace falta, valida las credenciales
    y descomprime el dataset en `dataset/melanoma_cancer_dataset/`.

    Requisitos previos:
      1. Tener cuenta en https://www.kaggle.com (gratis).
      2. Obtener el token API:
         Account -> Settings -> API -> "Create New Token"
         Se descarga un kaggle.json.
      3. Colocarlo en %USERPROFILE%\.kaggle\kaggle.json
         (la primera ejecución del script crea la carpeta si no existe).

.NOTES
    Tamaño: ~103 MB descomprimido. El dataset NO se versiona (está excluido en .gitignore).
#>

$ErrorActionPreference = 'Stop'

$repoRoot   = Split-Path -Parent $PSScriptRoot
$datasetDir = Join-Path $repoRoot 'dataset'
$kaggleJson = Join-Path $env:USERPROFILE '.kaggle\kaggle.json'

# 1. Verificar credenciales
if (-not (Test-Path $kaggleJson)) {
    Write-Host "`nFALTA kaggle.json" -ForegroundColor Red
    Write-Host "  1) Ve a https://www.kaggle.com/settings/account"
    Write-Host "  2) Pulsa 'Create New Token' en la seccion API"
    Write-Host "  3) Mueve el kaggle.json descargado a: $kaggleJson"
    Write-Host "  4) Vuelve a ejecutar este script.`n"
    exit 1
}

# 2. Verificar Python (requerido por el CLI de Kaggle)
$pyOk = $false
foreach ($cmd in @('python', 'py')) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found -and (& $cmd --version 2>$null)) { $pyOk = $true; $py = $cmd; break }
}
if (-not $pyOk) {
    Write-Host "Python no esta instalado. Instalalo desde https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# 3. Instalar kaggle CLI si falta
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando Kaggle CLI..." -ForegroundColor Cyan
    & $py -m pip install --user kaggle
}

# 4. Descargar
New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null
Write-Host "Descargando dataset (~103 MB)..." -ForegroundColor Cyan
kaggle datasets download `
    -d hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images `
    -p $datasetDir `
    --unzip

Write-Host "`nListo. Dataset en: $datasetDir" -ForegroundColor Green
Get-ChildItem $datasetDir | Format-Table -AutoSize
