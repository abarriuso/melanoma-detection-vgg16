#!/usr/bin/env python3
"""Detecta duplicados y cuasi-duplicados entre train y test del dataset local.

No borra ni modifica nada: solo genera `internal/dedup_report.json`.

Metodo
------
1. MD5 exacto del fichero de cada imagen de train y test.
2. Hash perceptual average-hash (aHash) 16x16 -> entero de 256 bits
   (imagen convertida a grises, reescalada a 16x16 con LANCZOS,
   bit = pixel >= media). Se implementa a mano con Pillow para no
   depender de `imagehash`.
3. Se comparan las ~1 000 imagenes de test contra las ~9 600 de train:
   - duplicado exacto: mismo MD5;
   - cuasi-duplicado: distancia de Hamming entre aHash <= UMBRAL_HAMMING
     (por defecto 12 bits de 256, el umbral usado en la auditoria 02;
     tambien se reportan recuentos con 8, 6 y 2 para sensibilidad).

Uso:
    python scripts/dedup_test.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "archive" / "melanoma_cancer_dataset"
REPORT = ROOT / "internal" / "dedup_report.json"

AHASH_SIZE = 16                # 16x16 -> 256 bits
UMBRAL_HAMMING = 12            # umbral principal (auditoria 02)
OTROS_UMBRALES = (8, 6, 2)     # recuentos de sensibilidad
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(path: Path) -> int:
    """Average-hash 16x16 como entero de 256 bits."""
    with Image.open(path) as im:
        im = im.convert("L").resize((AHASH_SIZE, AHASH_SIZE), Image.LANCZOS)
        px = list(im.getdata())
    mean = sum(px) / len(px)
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p >= mean else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def recolectar(split: str) -> list[dict]:
    entradas = []
    for clase in ("benign", "malignant"):
        carpeta = DATASET / split / clase
        for path in sorted(carpeta.iterdir()):
            if path.suffix.lower() not in EXTENSIONES:
                continue
            entradas.append(
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "split": split,
                    "clase": clase,
                    "md5": md5_file(path),
                    "ahash": ahash(path),
                }
            )
    return entradas


def main() -> int:
    t0 = time.time()
    if not DATASET.is_dir():
        print(f"ERROR: no existe {DATASET}", file=sys.stderr)
        return 1

    print("Procesando train...")
    train = recolectar("train")
    print(f"  {len(train)} imagenes ({time.time() - t0:.0f} s)")
    print("Procesando test...")
    test = recolectar("test")
    print(f"  {len(test)} imagenes ({time.time() - t0:.0f} s)")

    # --- Duplicados exactos (MD5) ---
    md5_train: dict[str, list[str]] = {}
    for e in train:
        md5_train.setdefault(e["md5"], []).append(e["path"])

    exactos = []  # pares test<->train con MD5 identico
    for e in test:
        for train_path in md5_train.get(e["md5"], []):
            exactos.append({"test": e["path"], "train": train_path, "md5": e["md5"]})

    # --- Cuasi-duplicados (aHash, Hamming) ---
    umbrales = (UMBRAL_HAMMING,) + tuple(u for u in OTROS_UMBRALES if u != UMBRAL_HAMMING)
    umbral_max = max(umbrales)
    # candidatos: distancia <= umbral_max; luego se re-clasifica por umbral
    pares_hash = []  # (test_path, train_path, distancia)
    for e in test:
        ht = e["ahash"]
        for tr in train:
            d = hamming(ht, tr["ahash"])
            if d <= umbral_max:
                pares_hash.append(
                    {"test": e["path"], "train": tr["path"], "hamming": d}
                )

    # Recuento de imagenes de test contaminadas por umbral
    contaminadas_por_umbral = {}
    for u in umbrales:
        tests_u = sorted({p["test"] for p in pares_hash if p["hamming"] <= u})
        # anadir tambien los exactos por MD5 (pueden escapar al hash si hubiera
        # metadatos distintos con pixels identicos ya los captura el aHash, pero
        # un MD5 exacto siempre cuenta como duplicado)
        tests_u = sorted(set(tests_u) | {p["test"] for p in exactos})
        contaminadas_por_umbral[str(u)] = {
            "n_pares": sum(1 for p in pares_hash if p["hamming"] <= u),
            "n_test_contaminadas": len(tests_u),
            "pct_test_contaminadas": round(100 * len(tests_u) / len(test), 1),
            "test_contaminadas": tests_u,
        }

    # Distribucion de distancias del mejor match de cada imagen de test
    mejor_dist = {}
    for e in test:
        ht = e["ahash"]
        mejor = min(hamming(ht, tr["ahash"]) for tr in train)
        mejor_dist[e["path"]] = mejor
    histograma = {}
    for d in mejor_dist.values():
        b = "0" if d == 0 else ("1-4" if d <= 4 else ("5-8" if d <= 8 else ("9-12" if d <= 12 else ">12")))
        histograma[b] = histograma.get(b, 0) + 1

    # Duplicados exactos intra-train (informativo: pueden cruzar el split train/val)
    grupos_intra_train = {
        m: paths for m, paths in md5_train.items() if len(paths) > 1
    }

    contaminadas_principal = contaminadas_por_umbral[str(UMBRAL_HAMMING)]

    reporte = {
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "metodo": {
            "exacto": "MD5 del fichero",
            "perceptual": f"aHash {AHASH_SIZE}x{AHASH_SIZE} ({AHASH_SIZE * AHASH_SIZE} bits), "
            "grises + resize LANCZOS, bit = pixel >= media",
            "umbral_hamming_principal": UMBRAL_HAMMING,
            "umbrales_sensibilidad": list(OTROS_UMBRALES),
            "nota": "Los duplicados exactos por MD5 se contabilizan siempre como "
            "contaminacion, independientemente del aHash.",
        },
        "n_train": len(train),
        "n_test": len(test),
        "exactos": {
            "n_pares": len(exactos),
            "n_test_contaminadas": len({p["test"] for p in exactos}),
            "pares": exactos,
        },
        "near_dupes_por_umbral": contaminadas_por_umbral,
        "histograma_mejor_match_test": dict(sorted(histograma.items())),
        "intra_train_md5_grupos": {
            "n_grupos": len(grupos_intra_train),
            "grupos": grupos_intra_train,
        },
        "test_contaminadas_umbral_principal": contaminadas_principal["test_contaminadas"],
        "n_test_contaminadas_umbral_principal": contaminadas_principal["n_test_contaminadas"],
        "duracion_segundos": round(time.time() - t0, 1),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nExactos MD5: {len(exactos)} pares "
          f"({len({p['test'] for p in exactos})} imagenes de test)")
    for u, datos in contaminadas_por_umbral.items():
        print(f"Hamming <= {u}: {datos['n_pares']} pares, "
              f"{datos['n_test_contaminadas']} test contaminadas "
              f"({datos['pct_test_contaminadas']} %)")
    print(f"Grupos de duplicados exactos intra-train: {len(grupos_intra_train)}")
    print(f"Informe: {REPORT}")
    print(f"Duracion total: {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
