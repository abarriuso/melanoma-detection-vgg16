#!/usr/bin/env python
"""
Convierte los .keras entrenados en Kaggle a TensorFlow.js (uint8) para la demo.

Replica exactamente la conversion que hace el notebook
(entrenamiento_conjunto_kaggle.ipynb), por si el entorno de Kaggle no tenia
`tensorflowjs` instalado y se salto ese paso.

No reentrena: solo carga cada .keras y lo exporta a demo/public/model/<id>/.

Uso (desde la raiz del repo):
    pip install "tensorflow>=2.15" "tensorflowjs>=4.22" tf-keras
    python scripts/convert_kaggle_output.py

Requiere que exista:
    models/_kaggle_output/melanoma_model/
        resultados_comparativa.json
        <id>/melanoma_<id>_final.keras   (vgg16, resnet50v2, efficientnetv2s)
"""

import os

# tensorflowjs aun no soporta Keras 3 de forma fiable: forzamos Keras 2 (legacy)
# ANTES de importar TensorFlow (igual que en el notebook).
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from datetime import datetime
from pathlib import Path

import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# El VGG16 lleva una capa Lambda con preprocess_input; hay que pasarla explicita.
CUSTOM_OBJECTS = {"preprocess_input": vgg16_preprocess}

# --- Rutas (relativas a la raiz del repo) ---
ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "models" / "_kaggle_output" / "melanoma_model"
DEMO_MODEL_DIR = ROOT / "demo" / "public" / "model"
RESULTS_JSON = KAGGLE_DIR / "resultados_comparativa.json"

VERSION = "2.0.0"  # nueva ejecucion -> nueva version (invalida la cache del cliente)

TARGET_LAYERS = {
    "vgg16": "block5_conv3",
    "resnet50v2": "post_relu",
    "efficientnetv2s": "top_conv",
}


def main():
    if not RESULTS_JSON.exists():
        raise SystemExit(f"No encuentro {RESULTS_JSON}. Descarga primero el output de Kaggle.")

    resultados = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    for model_id, target_layer in TARGET_LAYERS.items():
        keras_path = KAGGLE_DIR / model_id / f"melanoma_{model_id}_final.keras"
        if not keras_path.exists():
            print(f"[SKIP] {model_id}: no existe {keras_path}")
            continue

        temperature = float(resultados.get(model_id, {}).get("temperature", 1.0))
        out_dir = DEMO_MODEL_DIR / model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{model_id}] Cargando {keras_path.name} ...")
        # safe_mode=False + custom_objects: el VGG16 lleva una capa Lambda con
        # preprocess_input; son modelos propios y de confianza.
        model = load_model(keras_path, safe_mode=False, custom_objects=CUSTOM_OBJECTS)

        print(f"[{model_id}] Convirtiendo a TF.js (uint8), T = {temperature:.4f} ...")
        tfjs.converters.save_keras_model(
            model,
            str(out_dir),
            quantization_dtype_map={"uint8": "*"},
            metadata={
                "temperature": temperature,
                "version": VERSION,
                "modelId": model_id,
                "targetLayer": target_layer,
                "convertedAt": datetime.now().isoformat(),
            },
        )

        total_mb = sum(f.stat().st_size for f in out_dir.glob("*")) / 1024 / 1024
        print(f"[{model_id}] Listo -> {out_dir}  ({total_mb:.1f} MB)")

    print("\nConversion completada. Recuerda hacer commit de demo/public/model/*/")


if __name__ == "__main__":
    main()
