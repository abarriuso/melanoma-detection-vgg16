# =====================================================================
# CELDA PARA KAGGLE — Convierte los .keras entrenados a TensorFlow.js
# SIN reentrenar. Copia y pega esto en una celda de un notebook de Kaggle.
#
# Antes de ejecutar:
#   1. Crea un notebook nuevo en Kaggle (Create > New Notebook).
#   2. En el panel derecho: Add Input > Notebook Output (o "Your Work")
#      > busca y añade la salida del kernel  aadasss/mel-v2.
#      Se montara en /kaggle/input/<algo>/melanoma_model/...
#   3. Pega esta celda y ejecutala. Al terminar, descarga el output:
#        kaggle kernels output <tu-usuario>/<este-notebook> -p .\models\_tfjs
#      y copia model/<id>/* a  demo/public/model/<id>/  en el repo.
# =====================================================================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # los .keras se guardaron con Keras 2

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflowjs==4.22.0"], check=True)

import glob, json
from datetime import datetime
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# El VGG16 lleva una capa Lambda con preprocess_input; hay que pasarla explicita.
CUSTOM_OBJECTS = {"preprocess_input": vgg16_preprocess}

TARGET_LAYERS = {
    "efficientnetv2s": "top_conv",
    "resnet50v2": "post_relu",
    "vgg16": "block5_conv3",
}


def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


results_path = _find("resultados_comparativa.json")
resultados = json.load(open(results_path)) if results_path else {}
print("resultados_comparativa.json:", results_path)

OUT = "/kaggle/working/model"
for model_id, target_layer in TARGET_LAYERS.items():
    keras_path = _find(f"melanoma_{model_id}_final.keras")
    if not keras_path:
        print(f"[SKIP] {model_id}: no encontrado en /kaggle/input")
        continue

    temperature = float(resultados.get(model_id, {}).get("temperature", 1.0))
    out_dir = os.path.join(OUT, model_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{model_id}] cargando {keras_path}")
    # safe_mode=False + custom_objects: el VGG16 lleva una capa Lambda con
    # preprocess_input; son modelos propios y de confianza.
    model = load_model(keras_path, safe_mode=False, custom_objects=CUSTOM_OBJECTS)
    print(f"[{model_id}] convirtiendo a TF.js (uint8), T = {temperature:.4f}")
    tfjs.converters.save_keras_model(
        model, out_dir,
        quantization_dtype_map={"uint8": "*"},
        metadata={
            "temperature": temperature,
            "version": "2.0.0",
            "modelId": model_id,
            "targetLayer": target_layer,
            "convertedAt": str(datetime.now()),
        },
    )
    mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1024 / 1024
    print(f"[{model_id}] listo -> {out_dir}  ({mb:.1f} MB): {os.listdir(out_dir)}")

print("\nHecho. Descarga /kaggle/working/model/ y copialo a demo/public/model/")
