#!/usr/bin/env bash
# melanoma-detection-vgg16 — Watermarks Cleaner
# Limpia imágenes dermatoscópicas, notebooks y assets del demo

set -euo pipefail

WMR="C:/COSAS/PROYECTOS/watermarks-remover"
PYTHON="$WMR/.venv/Scripts/python.exe"
CLEAN_FILE="$PYTHON $WMR/service/scripts/clean_file.py"
CLEAN_IMAGE="$PYTHON $WMR/service/scripts/clean_image.py"
INSPECT="$PYTHON $WMR/service/scripts/inspect_file.py"

# Directorios a excluir
PRUNE="-name node_modules -o -name .git -o -name __pycache__ -o -name .venv"

echo "🧹 melanoma-detection-vgg16 Watermarks Cleaner"
echo "================================================"

# Limpiar imágenes de entrenamiento/validación
if [ -d "assets" ]; then
  echo ""
  echo "📁 Procesando assets/ (imágenes dermatoscópicas)..."
  find assets -type d \( $PRUNE \) -prune -o -type f \( \
    -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o \
    -iname '*.bmp' -o -iname '*.tiff' -o -iname '*.webp' \
  \) -print | head -1000 | while read -r file; do
    echo "  🔬 $file"
    $CLEAN_FILE "$file" -o "$file.cleaned" 2>/dev/null || true
    if [ -f "$file.cleaned" ]; then
      mv "$file.cleaned" "$file"
      echo "     ✅ Metadata limpia"
    fi
  done
fi

# Limpiar notebooks
if [ -d "notebooks" ]; then
  echo ""
  echo "📁 Procesando notebooks/..."
  find notebooks -type d \( $PRUNE \) -prune -o -type f -iname '*.ipynb' -print | while read -r file; do
    echo "  📓 $file"
    $CLEAN_FILE "$file" -o "$file.cleaned" 2>/dev/null || true
    if [ -f "$file.cleaned" ]; then
      mv "$file.cleaned" "$file"
      echo "     ✅ Limpio"
    fi
  done
fi

# Limpiar assets del demo (excluyendo node_modules)
if [ -d "demo" ]; then
  echo ""
  echo "📁 Procesando demo/..."
  find demo -type d \( $PRUNE \) -prune -o -type f \( \
    -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o \
    -iname '*.svg' -o -iname '*.html' -o -iname '*.js' -o -iname '*.css' \
  \) -print | while read -r file; do
    echo "  🖼️  $file"
    $CLEAN_FILE "$file" -o "$file.cleaned" 2>/dev/null || true
    if [ -f "$file.cleaned" ]; then
      mv "$file.cleaned" "$file"
      echo "     ✅ Limpio"
    fi
  done
fi

# Limpiar scripts
if [ -d "scripts" ]; then
  echo ""
  echo "📁 Procesando scripts/..."
  find scripts -type d \( $PRUNE \) -prune -o -type f -iname '*.py' -print | while read -r file; do
    echo "  🐍 $file"
    $CLEAN_FILE "$file" -o "$file.cleaned" 2>/dev/null || true
    if [ -f "$file.cleaned" ]; then
      mv "$file.cleaned" "$file"
      echo "     ✅ Limpio"
    fi
  done
fi

# Limpiar archive (limitado a 500 archivos para evitar tiempos excesivos)
if [ -d "archive" ]; then
  echo ""
  echo "📁 Procesando archive/ (máximo 500 archivos)..."
  find archive -type d \( $PRUNE \) -prune -o -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print | head -500 | while read -r file; do
    echo "  🗄️  $file"
    $CLEAN_FILE "$file" -o "$file.cleaned" 2>/dev/null || true
    if [ -f "$file.cleaned" ]; then
      mv "$file.cleaned" "$file"
      echo "     ✅ Limpio"
    fi
  done
fi

echo ""
echo "✅ melanoma-detection-vgg16 limpiado."
echo ""
echo "⚠️  Nota: Para imágenes médicas, la limpieza elimina metadata"
echo "   (EXIF, XMP, C2PA) pero NO altera los píxeles de la imagen."
