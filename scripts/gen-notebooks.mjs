// Genera 3 notebooks de entrenamiento (VGG16, ResNet50V2, EfficientNetV2S)
// listos para Google Colab con GPU T4.
//
//   node scripts/gen-notebooks.mjs   (desde la raíz del repo)
//
// Diseño:
//   - Cada notebook es AUTÓNOMO (no copia celdas de otro .ipynb).
//   - El modelo Keras acepta entrada en [0, 1] RGB (lo que entrega la demo
//     TF.js: resize -> /255) y hornea el preprocesado específico del backbone
//     dentro del grafo. Así el modelo exportado es self-contained.
//   - El backbone se construye con `input_tensor=` para obtener un grafo
//     plano (sin modelo anidado), lo que permite que Grad-CAM acceda a las
//     activaciones de la capa convolucional objetivo.
//   - Rigor clínico: métricas sens/spec, análisis de umbral, calibración
//     (temperature scaling), MC Dropout, Grad-CAM + Grad-CAM++, curva
//     Precision-Recall, TTA y análisis cualitativo de falsos negativos.

import { writeFileSync } from 'node:fs';

function cell(type, source) {
  const src = Array.isArray(source) ? source : [source];
  const c = { cell_type: type, metadata: {}, source: src };
  if (type === 'code') { c.outputs = []; c.execution_count = null; }
  return c;
}

// ──────────────────────────────────────────────────────────────
//  Celdas compartidas (idénticas en los 3 notebooks)
// ──────────────────────────────────────────────────────────────

function setupCell() {
  return cell('code', [
`# === Setup para Google Colab (GPU T4) ===
# La exportación a TF.js (tensorflowjs) todavía no soporta Keras 3 de forma
# fiable. Forzamos Keras 2 (legacy) en TODO el notebook: la variable debe
# fijarse ANTES de importar TensorFlow. Si Colab pide reiniciar el entorno
# tras la instalación, hazlo y reejecuta desde aquí.
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
!pip install -q tf-keras tensorflowjs`,
  ]);
}

function importsCell() {
  return cell('code', [
`import os, random, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    classification_report, roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, average_precision_score,
)
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import load_img, img_to_array

# Reproducibilidad
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow", tf.__version__, "| Keras", tf.keras.__version__)
assert tf.keras.__version__.startswith("2"), "Se requiere Keras 2 (legacy) para exportar a TF.js"`,
  ]);
}

function gpuCheckCell() {
  return cell('code', [
`# Comprobación de GPU (en Colab: Entorno de ejecución > Cambiar tipo > T4 GPU)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        print("GPU disponible:", g.name)
    # Crecimiento de memoria: evita que TF reserve toda la VRAM de golpe
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)
else:
    print("AVISO: no se detectó GPU. El entrenamiento será MUY lento.")
    print("En Colab: Entorno de ejecución > Cambiar tipo de entorno > T4 GPU.")`,
  ]);
}

function configCell(modeloId) {
  return cell('code', [
`from google.colab import drive
drive.mount("/content/drive")

# El dataset debe estar en Drive con la estructura:
#   MyDrive/melanoma_cancer_dataset/{train,test}/{benign,malignant}/*.jpg
BASE_DIR  = "/content/drive/MyDrive/melanoma_cancer_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")
TEST_BEN_DIR = os.path.join(TEST_DIR, "benign")
TEST_MAL_DIR = os.path.join(TEST_DIR, "malignant")
CLASS_NAMES  = ["benign", "malignant"]

MODELO_ID = "${modeloId}"
MODEL_DIR = "/content/drive/MyDrive/melanoma_model"
os.makedirs(os.path.join(MODEL_DIR, MODELO_ID), exist_ok=True)

# Hiperparámetros del pipeline
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
# Sesgo hacia la sensibilidad: penaliza más fallar un maligno (clase 1)
class_weight = {0: 1.0, 1: 1.3}

assert os.path.isdir(TRAIN_DIR), f"No existe {TRAIN_DIR}. ¿Subiste el dataset a Drive?"
assert os.path.isdir(TEST_DIR),  f"No existe {TEST_DIR}."
print("Dataset:", BASE_DIR)
print("Modelos:", os.path.join(MODEL_DIR, MODELO_ID))`,
  ]);
}

function dataCells() {
  return [
    cell('markdown', [
`## 1. Pipeline de datos con \`tf.data\`

Carga con \`image_dataset_from_directory\`, split 80/20 train/val (misma seed
para que sea complementario y sin solapamiento) y un conjunto de test
independiente. \`label_mode='binary'\` devuelve etiquetas float32 (0.0/1.0),
lo que espera \`binary_crossentropy\`.`,
    ]),
    cell('code', [
`AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
)
# shuffle=False en test: las predicciones mantienen el orden de los archivos
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False,
)

class_names = train_ds.class_names
assert class_names == CLASS_NAMES, f"Orden de clases inesperado: {class_names}"
print("Clases:", class_names)
print(f"Batches  train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")`,
    ]),
    cell('markdown', [
`### Augmentation y normalización

El modelo recibe la imagen en **[0, 1]** (igual que la demo TF.js, que hace
\`/255\`). El preprocesado propio de cada backbone se hornea dentro del modelo,
no aquí, para que el modelo exportado sea autocontenido.

Augmentation enriquecida: geométrica (flip/rotación/zoom/traslación) +
fotométrica (brillo/contraste), crítica en dermatoscopia donde la iluminación
varía entre dispositivos y centros. Solo se aplica en entrenamiento.`,
    ]),
    cell('code', [
`normalization = layers.Rescaling(1.0 / 255)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
    layers.RandomContrast(0.15),
], name="augmentation")

def preprocess_train(x, y):
    x = normalization(x)
    x = augmentation(x, training=True)
    return tf.clip_by_value(x, 0.0, 1.0), y

def preprocess_eval(x, y):
    return normalization(x), y

train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds.map(preprocess_eval,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds  = test_ds.map(preprocess_eval,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)`,
    ]),
    cell('code', [
`# Visualización de un batch tras augmentation
imgs, labels = next(iter(train_ds))
plt.figure(figsize=(12, 6))
for i in range(8):
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow(np.clip(imgs[i].numpy(), 0, 1))
    plt.title(CLASS_NAMES[int(labels[i].numpy()[0])])
    plt.axis("off")
plt.tight_layout(); plt.show()`,
    ]),
  ];
}

function helpersCell() {
  return cell('code', [
`def crear_callbacks(nombre_archivo, paciencia_es=7, paciencia_lr=3):
    """ModelCheckpoint (mejor val_loss) + EarlyStopping + ReduceLROnPlateau."""
    ruta = os.path.join(MODEL_DIR, MODELO_ID, nombre_archivo)
    return [
        ModelCheckpoint(ruta, monitor="val_loss", save_best_only=True,
                        mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=paciencia_es,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=paciencia_lr,
                          min_lr=1e-7, verbose=1),
    ]

def graficar_historial(history, titulo=""):
    """Dibuja accuracy y loss de entrenamiento/validación."""
    h = history.history
    clave = "accuracy" if "accuracy" in h else "acc"
    epocas = range(1, len(h[clave]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if titulo:
        fig.suptitle(titulo, fontsize=13)
    ax1.plot(epocas, h[clave], "r--", label="Entrenamiento")
    ax1.plot(epocas, h[f"val_{clave}"], "b", label="Validación")
    ax1.set_title("Precisión"); ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True)
    ax2.plot(epocas, h["loss"], "r--", label="Entrenamiento")
    ax2.plot(epocas, h["val_loss"], "b", label="Validación")
    ax2.set_title("Pérdida"); ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()
    print(f"Mejor val_accuracy: {max(h[f'val_{clave}']):.4f}")
    print(f"Mejor val_loss    : {min(h['val_loss']):.4f}")`,
  ]);
}

function phase1Cells(modeloNombre) {
  return [
    cell('markdown', ['## 3. Fase 1 — Extracción de características\n\nBackbone congelado: solo se entrena la cabeza nueva. RMSprop 1e-4.']),
    cell('code', [
`callbacks_f1 = crear_callbacks("phase1.keras", paciencia_es=7, paciencia_lr=3)

modelo.compile(optimizer=RMSprop(learning_rate=1e-4),
               loss="binary_crossentropy", metrics=["accuracy"])

history_f1 = modelo.fit(
    train_ds, validation_data=val_ds, epochs=20,
    class_weight=class_weight, callbacks=callbacks_f1, verbose=1,
)
graficar_historial(history_f1, "` + modeloNombre + ` — Fase 1")`,
    ]),
  ];
}

function evalCells() {
  return [
    cell('markdown', ['## 5. Evaluación sobre el conjunto de test']),
    cell('code', [
`test_loss, test_acc = modelo.evaluate(test_ds, verbose=1)
print(f"\\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

# Predicciones sobre todo el test (test_ds tiene shuffle=False)
y_true, y_pred_prob = [], []
for imgs, labels in test_ds:
    p = modelo.predict(imgs, verbose=0).flatten()
    y_true.extend(labels.numpy().flatten())
    y_pred_prob.extend(p)
y_true      = np.array(y_true).astype(int)
y_pred_prob = np.array(y_pred_prob)
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\\n" + "=" * 60)
print("REPORTE DE CLASIFICACIÓN (umbral 0.5)")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"AUC: {roc_auc:.4f}")
print(f"Matriz de confusión  TN={tn} FP={fp} | FN={fn} TP={tp}")`,
    ]),
    cell('code', [
`fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
ax1.plot([0, 1], [0, 1], "k--", label="Azar")
ax1.set_xlabel("Ratio Falso Positivo (FPR)")
ax1.set_ylabel("Recall — Verdadero Positivo (TPR)")
ax1.set_title("Curva ROC"); ax1.legend(); ax1.grid(True)
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Matriz de confusión")
plt.tight_layout(); plt.show()
print(f"Falsos negativos (maligno→benigno): {fn}  ← error clínico más grave")
print(f"Falsos positivos (benigno→maligno): {fp}")`,
    ]),

    // 6. Métricas clínicas
    cell('markdown', ['## 6. Métricas clínicas y robustez']),
    cell('markdown', ['### 6.1 Sensibilidad y especificidad']),
    cell('code', [
`sensibilidad  = tp / (tp + fn)
especificidad = tn / (tn + fp)
vpp = tp / (tp + fp)   # valor predictivo positivo (precisión maligno)
vpn = tn / (tn + fn)   # valor predictivo negativo
print("MÉTRICAS CLÍNICAS (umbral 0.5)")
print(f"  Sensibilidad (recall maligno): {sensibilidad:.3f}  -> {tp}/{tp+fn} melanomas detectados")
print(f"  Especificidad               : {especificidad:.3f}  -> {tn}/{tn+fp} benignos descartados")
print(f"  Valor predictivo positivo   : {vpp:.3f}")
print(f"  Valor predictivo negativo   : {vpn:.3f}")
print(f"  Melanomas NO detectados (FN): {fn}  (el error más grave)")`,
    ]),
    cell('markdown', ['### 6.2 Análisis del umbral de decisión']),
    cell('code', [
`# El umbral 0.5 es arbitrario. En clínica interesa maximizar la sensibilidad
# (no perder melanomas) aunque baje la especificidad (más biopsias).
umbrales = np.arange(0.05, 0.96, 0.05)
sens_list, spec_list = [], []
for t in umbrales:
    tn_, fp_, fn_, tp_ = confusion_matrix(y_true, (y_pred_prob >= t).astype(int)).ravel()
    sens_list.append(tp_ / (tp_ + fn_))
    spec_list.append(tn_ / (tn_ + fp_))

plt.figure(figsize=(9, 5))
plt.plot(umbrales, sens_list, "r-o", label="Sensibilidad")
plt.plot(umbrales, spec_list, "b-o", label="Especificidad")
plt.axvline(0.5, color="gray", ls="--", label="Umbral por defecto (0.5)")
plt.axvline(0.3, color="green", ls=":", label="Umbral clínico sugerido (0.3)")
plt.xlabel("Umbral"); plt.ylabel("Métrica")
plt.title("Trade-off sensibilidad / especificidad"); plt.legend(); plt.grid(True)
plt.show()

print("Umbral | Sensib. | Especif. | Melanomas perdidos")
for t in [0.5, 0.4, 0.3, 0.2]:
    tn_, fp_, fn_, tp_ = confusion_matrix(y_true, (y_pred_prob >= t).astype(int)).ravel()
    print(f"  {t:.2f} |  {tp_/(tp_+fn_):.3f}  |  {tn_/(tn_+fp_):.3f}   |   {fn_}")`,
    ]),
    cell('markdown', ['### 6.3 Calibración (Temperature Scaling)']),
    cell('code', [
`# Ajusta una temperatura T sobre los logits del set de validación para que la
# probabilidad sea honesta (una predicción del 80% acierte ~80% de las veces).
val_logits, val_labels = [], []
for imgs, labels in val_ds:
    p = modelo.predict(imgs, verbose=0).flatten()
    p = np.clip(p, 1e-7, 1 - 1e-7)
    val_logits.extend(np.log(p / (1 - p)))
    val_labels.extend(labels.numpy().flatten())
val_logits = np.array(val_logits)
val_labels = np.array(val_labels)

def _nll(t):
    cal = 1 / (1 + np.exp(-val_logits / t))
    return -np.mean(val_labels * np.log(cal + 1e-7) + (1 - val_labels) * np.log(1 - cal + 1e-7))

T_opt = float(minimize_scalar(_nll, bounds=(0.05, 10), method="bounded").x)

def _ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return ece

p_raw = 1 / (1 + np.exp(-val_logits))
p_cal = 1 / (1 + np.exp(-val_logits / T_opt))
print(f"Temperatura óptima T = {T_opt:.4f}")
print(f"ECE antes : {_ece(p_raw, val_labels):.4f}")
print(f"ECE tras  : {_ece(p_cal, val_labels):.4f}")
print(f"\\nCopia en demo/src/lib/constants.js -> temperature: {T_opt:.3f}")`,
    ]),
    cell('markdown', ['### 6.4 Incertidumbre (Monte Carlo Dropout)']),
    cell('code', [
`# N inferencias con Dropout ACTIVO (training=True). La desviación típica mide
# cuánto "duda" el modelo: alta sigma -> candidata a derivar a especialista.
def mc_dropout_predict(arr, n_iter=30):
    preds = np.array([modelo(arr, training=True).numpy().ravel()[0] for _ in range(n_iter)])
    return preds.mean(), preds.std()

ejemplos = ([os.path.join(TEST_MAL_DIR, f) for f in sorted(os.listdir(TEST_MAL_DIR))[:6]] +
            [os.path.join(TEST_BEN_DIR, f) for f in sorted(os.listdir(TEST_BEN_DIR))[:6]])
random.shuffle(ejemplos)

fig, axs = plt.subplots(3, 4, figsize=(16, 12))
for ax, ruta in zip(axs.flatten(), ejemplos):
    img = load_img(ruta, target_size=(224, 224))
    arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    media, sigma = mc_dropout_predict(arr)
    clase = "Maligno" if media >= 0.5 else "Benigno"
    conf, color = ("ALTA", "green") if sigma < 0.10 else (("MEDIA", "orange") if sigma < 0.20 else ("BAJA", "red"))
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{clase} {media*100:.0f}% (±{sigma*100:.0f}%)\\nfiabilidad: {conf}", color=color, fontsize=11)
plt.suptitle("Monte Carlo Dropout — 30 pasadas", fontsize=14)
plt.tight_layout(); plt.show()`,
    ]),

    // 7. Grad-CAM
    cell('markdown', ['## 7. Interpretabilidad — Grad-CAM y Grad-CAM++\n\nVerifican que el modelo mira la lesión y no artefactos (pelo, reglas, reflejos).']),
    cell('code', [
`def make_gradcam_heatmap(arr):
    """Grad-CAM sobre TARGET_LAYER. Devuelve (heatmap, score de maligno)."""
    x = tf.constant(arr, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_cam_model(x)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_mean(conv_out[0] * pooled, axis=-1))
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(score.numpy()[0])

def make_gradcampp_heatmap(arr):
    """Grad-CAM++: pondera gradientes positivos píxel a píxel (mejor localización)."""
    x = tf.constant(arr, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_cam_model(x)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)[0]
    conv = conv_out[0]
    grads_2, grads_3 = grads ** 2, grads ** 3
    denom = 2.0 * grads_2 + tf.reduce_sum(conv, axis=(0, 1)) * grads_3 + 1e-8
    alphas = grads_2 / denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), float(score.numpy()[0])

def comparar_gradcam(img_path, clase_esperada=None):
    img = load_img(img_path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    batch = np.expand_dims(arr, axis=0)
    hm, pred = make_gradcam_heatmap(batch)
    hmpp, _ = make_gradcampp_heatmap(batch)
    def overlay(h):
        h = tf.image.resize(h[..., np.newaxis], (224, 224)).numpy().squeeze()
        return np.clip(0.4 * plt.cm.jet(h)[:, :, :3] + 0.6 * arr, 0, 1)
    clase = "Maligno" if pred >= 0.5 else "Benigno"
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    for a, im, t in zip(ax, [arr, overlay(hm), overlay(hmpp)], ["Original", "Grad-CAM", "Grad-CAM++"]):
        a.imshow(im); a.set_title(t); a.axis("off")
    titulo = f"Pred: {clase} ({pred*100:.0f}%)"
    if clase_esperada:
        titulo += f"  |  Real: {clase_esperada}"
    fig.suptitle(titulo, fontsize=13); plt.tight_layout(); plt.show()

for f in sorted(os.listdir(TEST_MAL_DIR))[:3]:
    comparar_gradcam(os.path.join(TEST_MAL_DIR, f), clase_esperada="Maligno")`,
    ]),

    // 8. PR curve
    cell('markdown', ['## 8. Curva Precision-Recall']),
    cell('code', [
`# La ROC es informativa con clases balanceadas (este test es 50/50), pero en
# clínica real los malignos son minoría y la PR es más discriminativa.
prec, rec, _ = precision_recall_curve(y_true, y_pred_prob)
ap = average_precision_score(y_true, y_pred_prob)
baseline = np.mean(y_true)
plt.figure(figsize=(7, 6))
plt.plot(rec, prec, lw=2, color="#0d9488", label=f"Modelo (AP = {ap:.3f})")
plt.axhline(baseline, ls="--", color="gray", label=f"Línea base (prevalencia = {baseline:.2f})")
plt.xlabel("Recall (sensibilidad)"); plt.ylabel("Precision (VPP)")
plt.title("Curva Precision-Recall"); plt.xlim([0, 1]); plt.ylim([0, 1.02])
plt.legend(loc="lower left"); plt.grid(True, alpha=0.3); plt.show()
print(f"Average Precision: {ap:.4f} | AUC ROC: {roc_auc:.4f}")`,
    ]),

    // 9. TTA
    cell('markdown', ['## 9. Test-Time Augmentation (TTA)']),
    cell('code', [
`# Promediamos predicciones de varias augmentaciones deterministas de la misma
# imagen (flips, rotaciones). Más robusto a artefactos de captura sin reentrenar.
def tta_predict(arr):
    augs = [arr, arr[:, :, ::-1, :], arr[:, ::-1, :, :], arr[:, ::-1, ::-1, :]]
    return np.mean([modelo.predict(a, verbose=0)[0][0] for a in augs])

print("Evaluando TTA sobre el test (puede tardar)...")
y_pred_tta = []
for x_batch, _ in test_ds:
    for img in x_batch.numpy():
        y_pred_tta.append(tta_predict(np.expand_dims(img, 0)))
y_pred_tta = np.array(y_pred_tta)
y_pred_tta_bin = (y_pred_tta >= 0.5).astype(int)

acc_simple = (y_pred == y_true).mean()
acc_tta = (y_pred_tta_bin == y_true).mean()
fn_simple = int(((y_true == 1) & (y_pred == 0)).sum())
fn_tta = int(((y_true == 1) & (y_pred_tta_bin == 0)).sum())
print(f"{'Métrica':<28}{'Simple':>10}{'TTA':>10}")
print(f"{'Accuracy':<28}{acc_simple:>10.4f}{acc_tta:>10.4f}")
print(f"{'Falsos negativos':<28}{fn_simple:>10}{fn_tta:>10}")
print(f"{'AUC':<28}{roc_auc:>10.4f}{auc(*roc_curve(y_true, y_pred_tta)[:2]):>10.4f}")`,
    ]),

    // 10. Error analysis
    cell('markdown', ['## 10. Análisis cualitativo de falsos negativos']),
    cell('code', [
`# Los FN son los errores clínicamente más graves (melanoma no detectado).
fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
print(f"Total FN: {len(fn_idx)} | score medio: {y_pred_prob[fn_idx].mean():.3f}" if len(fn_idx) else "Sin FN")

# Reconstruimos las rutas en el orden de test_ds (benignos primero, luego malignos)
paths_test = ([os.path.join(TEST_BEN_DIR, f) for f in sorted(os.listdir(TEST_BEN_DIR))] +
              [os.path.join(TEST_MAL_DIR, f) for f in sorted(os.listdir(TEST_MAL_DIR))])
if len(fn_idx):
    orden = fn_idx[np.argsort(y_pred_prob[fn_idx])][:6]   # los más "convencidos" de su error
    fig, axs = plt.subplots(2, len(orden), figsize=(3.3 * len(orden), 7))
    axs = np.atleast_2d(axs)
    for col, i in enumerate(orden):
        img = load_img(paths_test[i], target_size=(224, 224))
        arr = img_to_array(img) / 255.0
        hm, _ = make_gradcampp_heatmap(np.expand_dims(arr, 0))
        hm = tf.image.resize(hm[..., np.newaxis], (224, 224)).numpy().squeeze()
        overlay = np.clip(0.4 * plt.cm.jet(hm)[:, :, :3] + 0.6 * arr, 0, 1)
        axs[0, col].imshow(arr); axs[0, col].set_title(f"Score {y_pred_prob[i]:.3f}\\nReal: Maligno", color="red", fontsize=9); axs[0, col].axis("off")
        axs[1, col].imshow(overlay); axs[1, col].set_title("Atención", fontsize=9); axs[1, col].axis("off")
    plt.suptitle('Top falsos negativos (modelo más "convencido" de su error)', fontsize=13)
    plt.tight_layout(); plt.show()`,
    ]),

    // 11. Save + convert
    cell('markdown', ['## 11. Guardar y convertir a TF.js']),
    cell('code', [
`ruta_keras = os.path.join(MODEL_DIR, MODELO_ID, f"melanoma_{MODELO_ID}_final.keras")
modelo.save(ruta_keras)
print("Modelo Keras guardado:", ruta_keras)`,
    ]),
    cell('code', [
`import tensorflowjs as tfjs   # instalado en la celda de setup

TEMPERATURE = float(T_opt)   # de la celda de calibración (6.3)
converted_dir = os.path.join(MODEL_DIR, MODELO_ID, "tfjs")
os.makedirs(converted_dir, exist_ok=True)

tfjs.converters.save_keras_model(
    modelo, converted_dir,
    quantization_dtype_map={"uint8": "*"},
    metadata={
        "temperature": TEMPERATURE,
        "version": "1.0.0",
        "modelId": MODELO_ID,
        "targetLayer": TARGET_LAYER,
        "convertedAt": str(datetime.now()),
    },
)
print("TF.js guardado en:", converted_dir)
for f in sorted(os.listdir(converted_dir)):
    print(" ", f)`,
    ]),
    cell('markdown', [
`---
### Conversión completada

1. Descarga la carpeta \`tfjs/\` desde Drive y copia su contenido a
   \`demo/public/model/${'$'}{MODELO_ID}/\` en el repo.
2. Actualiza \`temperature\` (y, tras entrenar, las métricas AUC/sens/spec) del
   modelo en \`demo/src/lib/constants.js\`.`,
    ]),
  ];
}

// ──────────────────────────────────────────────────────────────
//  Builder de un notebook completo
// ──────────────────────────────────────────────────────────────

function buildNotebook(cfg) {
  const cells = [];
  cells.push(cell('markdown', [cfg.titleMd]));
  cells.push(setupCell());
  cells.push(importsCell());
  cells.push(gpuCheckCell());
  cells.push(configCell(cfg.id));
  cells.push(...dataCells());
  cells.push(cell('markdown', [`## 2. Arquitectura — ${cfg.name}\n\n${cfg.archNote}`]));
  cells.push(cell('code', [cfg.archCode]));
  cells.push(helpersCell());
  cells.push(...phase1Cells(cfg.name));
  cells.push(cell('markdown', [`## 4. Fase 2 — Fine-tuning\n\n${cfg.phase2Note}`]));
  cells.push(cell('code', [cfg.phase2Freeze]));
  cells.push(cell('code', [
`modelo.compile(optimizer=Adam(learning_rate=${cfg.lrF2}),
               loss="binary_crossentropy", metrics=["accuracy"])
callbacks_f2 = crear_callbacks("finetuning.keras", paciencia_es=${cfg.esF2}, paciencia_lr=${cfg.lrPat})
history_f2 = modelo.fit(
    train_ds, validation_data=val_ds, epochs=${cfg.epochsF2},
    class_weight=class_weight, callbacks=callbacks_f2, verbose=1,
)
graficar_historial(history_f2, "${cfg.name} — Fine-tuning")`,
  ]));
  cells.push(...evalCells());

  return {
    nbformat: 4,
    nbformat_minor: 0,
    metadata: {
      colab: { provenance: [], gpuType: 'T4', toc_visible: true },
      kernelspec: { name: 'python3', display_name: 'Python 3' },
      language_info: { name: 'python' },
      accelerator: 'GPU',
      title: `${cfg.name} — Melanoma Detection`,
    },
    cells,
  };
}

// ──────────────────────────────────────────────────────────────
//  Configuraciones de los 3 modelos
// ──────────────────────────────────────────────────────────────

const VGG16 = {
  id: 'vgg16',
  name: 'VGG16',
  titleMd:
`# Detección de melanomas — VGG16

**Clasificación binaria de lesiones dermatoscópicas (benigno / maligno)**

Adrián Barriuso Pizarro · 2026

---

| Arquitectura | Detalle |
|---|---|
| Backbone | VGG16 (ImageNet) + GAP → Dense(256,ReLU) → Dropout(0.5) → Dense(1,Sigmoid) |
| Entrada | [0, 1] RGB 224×224 (preprocesado horneado en el modelo) |
| Dataset | Melanoma Skin Cancer (10 000 imágenes, Kaggle, CC0) |
| Entorno | Google Colab · GPU T4 |
| Seed | 42 |

**Fase 1** — VGG16 congelado, RMSprop 1e-4, 20 epochs.
**Fase 2** — Bloque 5 descongelado (últimas 4 capas), Adam 1e-5, 30 epochs.`,
  archNote:
`VGG16 se alimenta directamente con la imagen en [0, 1] (mismo preprocesado que
la demo y que el modelo desplegado). Se construye con \`input_tensor=\` para
obtener un grafo plano y poder aplicar Grad-CAM sobre \`block5_conv3\`.`,
  archCode:
`from tensorflow.keras.applications import VGG16

TARGET_LAYER = "block5_conv3"   # última conv de VGG16 (14×14×512)

inputs = Input(shape=(224, 224, 3))   # [0, 1] RGB
base = VGG16(input_tensor=inputs, include_top=False, weights="imagenet")
base.trainable = False                # Fase 1: congelado

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
modelo = Model(inputs, outputs, name="melanoma_vgg16")

# Grad-CAM: misma red, dos salidas (activaciones conv + predicción). Grafo plano.
grad_cam_model = Model(inputs, [base.get_layer(TARGET_LAYER).output, modelo.output])

modelo.summary()
ent = sum(int(tf.size(w)) for w in modelo.trainable_weights)
tot = sum(int(tf.size(w)) for w in modelo.weights)
print(f"Parámetros totales: {tot:,} | entrenables: {ent:,}")`,
  phase2Note:
`Descongelamos las últimas 4 capas (block5_conv1-3 + block5_pool). LR bajo
(Adam 1e-5) para adaptar features de alto nivel sin destruir las genéricas.`,
  phase2Freeze:
`base.trainable = True
for layer in base.layers[:-4]:
    layer.trainable = False
entrenables = [l.name for l in base.layers if l.trainable]
print("Capas entrenables del backbone:", entrenables)`,
  lrF2: '1e-5', epochsF2: 30, esF2: 10, lrPat: 4,
};

const RESNET = {
  id: 'resnet50v2',
  name: 'ResNet50V2',
  titleMd:
`# Detección de melanomas — ResNet50V2

**Clasificación binaria de lesiones dermatoscópicas (benigno / maligno)**

Adrián Barriuso Pizarro · 2026

---

| Arquitectura | Detalle |
|---|---|
| Backbone | ResNet50V2 (ImageNet) + GAP → Dense(256,ReLU) → Dropout(0.5) → Dense(1,Sigmoid) |
| Entrada | [0, 1] RGB 224×224 (reescala a [-1, 1] dentro del modelo) |
| Dataset | Melanoma Skin Cancer (10 000 imágenes, Kaggle, CC0) |
| Entorno | Google Colab · GPU T4 |
| Seed | 42 |

**Fase 1** — Backbone congelado, RMSprop 1e-4, 20 epochs.
**Fase 2** — ~50% de capas no-BN descongeladas, **BatchNorm congelado**,
Adam 1e-6 (más conservador: ResNet tiene más capacidad), 40 epochs.`,
  archNote:
`ResNet50V2 espera entrada en [-1, 1]; añadimos \`Rescaling(2, offset=-1)\` dentro
del modelo (la demo entrega [0, 1]). \`base(training=False)\` no aplica aquí
porque usamos \`input_tensor\`; el control de BatchNorm se hace en la Fase 2.`,
  archCode:
`from tensorflow.keras.applications import ResNet50V2

TARGET_LAYER = "post_relu"   # activación final de ResNet50V2 (7×7×2048)

inputs = Input(shape=(224, 224, 3))            # [0, 1] RGB
x = layers.Rescaling(2.0, offset=-1.0)(inputs) # -> [-1, 1]
base = ResNet50V2(input_tensor=x, include_top=False, weights="imagenet")
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
modelo = Model(inputs, outputs, name="melanoma_resnet50v2")

grad_cam_model = Model(inputs, [base.get_layer(TARGET_LAYER).output, modelo.output])

modelo.summary()
ent = sum(int(tf.size(w)) for w in modelo.trainable_weights)
tot = sum(int(tf.size(w)) for w in modelo.weights)
print(f"Parámetros totales: {tot:,} | entrenables: {ent:,}")`,
  phase2Note:
`Descongelamos ~50% de las capas no-BN. **BatchNorm permanece congelado** para
que sus estadísticas moving_mean/variance no se corrompan con los batches
pequeños del fine-tuning (problema clásico de ResNet). LR 1e-6.`,
  phase2Freeze:
`base.trainable = True
# Congelar TODAS las capas BatchNorm
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
# Descongelar las últimas ~80 capas no-BN
for layer in base.layers[-80:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True
bn_train = sum(1 for l in base.layers if isinstance(l, layers.BatchNormalization) and l.trainable)
no_bn = sum(1 for l in base.layers if not isinstance(l, layers.BatchNormalization) and l.trainable)
print(f"Entrenables no-BN: {no_bn} | BatchNorm entrenables: {bn_train} (debe ser 0)")`,
  lrF2: '1e-6', epochsF2: 40, esF2: 12, lrPat: 5,
};

const EFFNET = {
  id: 'efficientnetv2s',
  name: 'EfficientNetV2S',
  titleMd:
`# Detección de melanomas — EfficientNetV2S

**Clasificación binaria de lesiones dermatoscópicas (benigno / maligno)**

Adrián Barriuso Pizarro · 2026

---

| Arquitectura | Detalle |
|---|---|
| Backbone | EfficientNetV2S (ImageNet) + GAP → Dense(256,ReLU) → Dropout(0.5) → Dense(1,Sigmoid) |
| Entrada | [0, 1] RGB 224×224 (reescala a [0, 255] y usa el preprocesado interno) |
| Dataset | Melanoma Skin Cancer (10 000 imágenes, Kaggle, CC0) |
| Entorno | Google Colab · GPU T4 |
| Seed | 42 |

Mejor ratio accuracy/parámetros de su familia (~22 M, ~15 MB en TF.js cuantizado).

**Fase 1** — Backbone congelado, RMSprop 1e-4, 20 epochs.
**Fase 2** — ~50% de capas descongeladas, **BatchNorm congelado**, Adam 1e-5, 40 epochs.`,
  archNote:
`EfficientNetV2S incluye su propio preprocesado (\`include_preprocessing=True\`)
que espera [0, 255]. Como la demo entrega [0, 1], reescalamos con
\`Rescaling(255)\` justo antes del backbone y dejamos que Keras aplique la
normalización oficial. Así no dependemos de constantes internas.`,
  archCode:
`from tensorflow.keras.applications import EfficientNetV2S

TARGET_LAYER = "top_conv"   # última conv de EfficientNetV2S (7×7×1280)

inputs = Input(shape=(224, 224, 3))     # [0, 1] RGB
x = layers.Rescaling(255.0)(inputs)     # -> [0, 255] (lo que espera el preproc interno)
base = EfficientNetV2S(input_tensor=x, include_top=False, weights="imagenet",
                       include_preprocessing=True)
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
modelo = Model(inputs, outputs, name="melanoma_efficientnetv2s")

grad_cam_model = Model(inputs, [base.get_layer(TARGET_LAYER).output, modelo.output])

modelo.summary()
ent = sum(int(tf.size(w)) for w in modelo.trainable_weights)
tot = sum(int(tf.size(w)) for w in modelo.weights)
print(f"Parámetros totales: {tot:,} | entrenables: {ent:,}")`,
  phase2Note:
`Descongelamos ~50% de las capas. **BatchNorm congelado** (los bloques MBConv de
EfficientNetV2 también lo usan). LR 1e-5.`,
  phase2Freeze:
`base.trainable = True
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
mid = len(base.layers) // 2
for layer in base.layers[mid:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True
bn_train = sum(1 for l in base.layers if isinstance(l, layers.BatchNormalization) and l.trainable)
no_bn = sum(1 for l in base.layers if not isinstance(l, layers.BatchNormalization) and l.trainable)
print(f"Entrenables no-BN: {no_bn} | BatchNorm entrenables: {bn_train} (debe ser 0)")`,
  lrF2: '1e-5', epochsF2: 40, esF2: 12, lrPat: 5,
};

for (const cfg of [VGG16, RESNET, EFFNET]) {
  const nb = buildNotebook(cfg);
  const path = `notebooks/${cfg.id}.ipynb`;
  writeFileSync(path, JSON.stringify(nb, null, 1));
  console.log(`Wrote ${path} (${nb.cells.length} cells)`);
}

console.log('\nAll notebooks generated.');
