#!/usr/bin/env python3
"""Decodifica los JPEG del test a PPM (P6, RGB crudo) con Pillow/libjpeg-turbo.

Motivo: `jpeg-js` (decodificador JS puro, upsampling de croma por caja)
desvia las metricas respecto a la evaluacion original de Kaggle
(TF decode_jpeg, libjpeg). Pillow usa libjpeg-turbo como TF, asi que la
decodificacion coincide y el resize bilinear se hace despues en tfjs,
exactamente como en la demo y en el entrenamiento.

Salida: internal/reval/ppm/<split>/{benign,malignant}/*.ppm
Uso: python scripts/decode_test_ppm.py [split=test] [limite_por_clase=0]
"""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = todas (por clase)
    src = ROOT / "archive" / "melanoma_cancer_dataset" / split
    dst = ROOT / "internal" / "reval" / "ppm" / split
    n = 0
    for clase in ("benign", "malignant"):
        out_dir = dst / clase
        out_dir.mkdir(parents=True, exist_ok=True)
        ficheros = sorted((src / clase).iterdir())
        if limit:
            ficheros = ficheros[:limit]
        for path in ficheros:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            out = out_dir / (path.stem + ".ppm")
            if out.exists():
                continue
            with Image.open(path) as im:
                im.convert("RGB").save(out, format="PPM")
            n += 1
    print(f"Decodificadas {n} imagenes nuevas en {dst}")


if __name__ == "__main__":
    main()
