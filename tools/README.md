# 🧹 Watermarks Cleaner

Script integrado para limpiar marcas de proveniencia AI, metadata y caracteres Unicode invisibles de los archivos de este proyecto.

## Uso

```bash
# Desde la raíz del proyecto
./tools/clean-watermarks.sh
```

## Qué limpia

- Imágenes dermatoscópicas (`assets/`)
- Notebooks (`notebooks/`)
- Assets del demo (`demo/`)
- Scripts Python (`scripts/`)
- Archivos del modelo (`archive/`)
- Metadata C2PA, EXIF, XMP (sin alterar píxeles)

## ⚠️ Nota importante

Para imágenes médicas, la limpieza **elimina metadata** (EXIF, XMP, C2PA) de las imágenes dermatoscópicas pero **NO altera los píxeles**. La información clínica contenida en los píxeles se preserva.

## Requisito

Tener clonado `watermarks-remover` en:
```
C:\COSAS\PROYECTOS\watermarks-remover
```

Más info: [GUÍA_RÁPIDA.md](../watermarks-remover/GUÍA_RÁPIDA.md)
