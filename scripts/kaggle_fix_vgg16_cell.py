# =====================================================================
# CELDA KAGGLE — Reconvierte VGG16 a TF.js SIN capa Lambda.
#
# Problema: build_vgg16() del notebook mete el preprocesado Caffe en una capa
# `Lambda(lambda t: preprocess_input(t*255))`. TensorFlow.js NO sabe deserializar
# una Lambda con funcion Python, asi que el vgg16/model.json actual NO carga en
# el navegador.
#
# Solucion: reconstruir VGG16 con el mismo preprocesado pero como capas nativas
# que TF.js si soporta:
#   - Rescaling(255)            -> [0,1] a [0,255] (RGB)
#   - Conv2D 1x1 fijo           -> RGB->BGR + resta de medias ImageNet (Caffe)
# y copiar los pesos entrenados desde el .keras original (por nombre de capa).
#
# Requisitos: mismo notebook con el output de aadasss/mel-v2 como input.
# Pega esta celda DESPUES de haber ejecutado la conversion normal y ejecutala.
# =====================================================================

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import glob, json
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + \
           glob.glob(f"/kaggle/working/**/{name}", recursive=True)
    return hits[0] if hits else None


def build_vgg16_native():
    """VGG16 con preprocesado Caffe horneado como capas nativas (sin Lambda)."""
    inputs = Input(shape=(224, 224, 3))          # [0, 1] RGB (igual que la demo)
    x = layers.Rescaling(255.0)(inputs)          # -> [0, 255] RGB
    # Conv2D 1x1 fijo: permuta RGB->BGR y resta las medias Caffe.
    x = layers.Conv2D(3, 1, use_bias=True, name="caffe_preproc")(x)
    base = VGG16(input_tensor=x, include_top=False, weights=None)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="melanoma_vgg16")


tf.keras.backend.clear_session()

kp = find("melanoma_vgg16_final.keras")
rj = find("resultados_comparativa.json")
T = float(json.load(open(rj)).get("vgg16", {}).get("temperature", 1.0)) if rj else 1.0
assert kp, "No encuentro melanoma_vgg16_final.keras"

# 1) Modelo original (con Lambda) solo para extraer pesos.
orig = load_model(kp, safe_mode=False,
                  custom_objects={"preprocess_input": preprocess_input})
orig_by_name = {l.name: l for l in orig.layers}

# 2) Modelo nativo equivalente.
native = build_vgg16_native()

# 3) Pesos fijos del preprocesado Caffe (RGB[R,G,B] -> BGR[B,G,R] - medias).
k = np.zeros((1, 1, 3, 3), np.float32)
k[0, 0, 2, 0] = 1.0   # out0 (B) <- in2 (B)
k[0, 0, 1, 1] = 1.0   # out1 (G) <- in1 (G)
k[0, 0, 0, 2] = 1.0   # out2 (R) <- in0 (R)
b = -np.array([103.939, 116.779, 123.68], np.float32)  # medias Caffe (BGR)
native.get_layer("caffe_preproc").set_weights([k, b])
native.get_layer("caffe_preproc").trainable = False

# 4) Copiar el resto de pesos entrenados por nombre de capa.
copiadas, faltan = 0, []
for layer in native.layers:
    if layer.name == "caffe_preproc" or not layer.get_weights():
        continue
    src = orig_by_name.get(layer.name)
    if src is None:
        faltan.append(layer.name); continue
    try:
        layer.set_weights(src.get_weights()); copiadas += 1
    except Exception as e:
        faltan.append(f"{layer.name} ({e})")
print(f"Capas con pesos copiados: {copiadas} | sin copiar: {faltan}")

# 5) Comprobacion rapida: misma prediccion que el original en una imagen dummy.
x = np.random.rand(1, 224, 224, 3).astype("float32")
d = float(np.abs(orig.predict(x, verbose=0) - native.predict(x, verbose=0)).max())
print(f"Diferencia maxima orig vs nativo: {d:.6f}  (debe ser ~0)")
assert d < 1e-3, "Las predicciones NO coinciden; revisar antes de desplegar."

# 6) Convertir a TF.js (uint8) sobre demo/public/model/vgg16.
outdir = "/kaggle/working/model/vgg16"
os.makedirs(outdir, exist_ok=True)
tfjs.converters.save_keras_model(
    native, outdir,
    quantization_dtype_map={"uint8": "*"},
    metadata={"temperature": T, "version": "2.0.0", "modelId": "vgg16",
              "targetLayer": "block5_conv3", "convertedAt": str(datetime.now())},
)
print("vgg16 (sin Lambda) OK ->", os.listdir(outdir))
