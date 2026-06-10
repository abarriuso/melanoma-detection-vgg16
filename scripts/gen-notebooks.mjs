// Generates 3 training notebooks for VGG16, ResNet50V2, EfficientNetV2S.
// Run: node scripts/gen-notebooks.mjs  (from repo root)

import { readFileSync, writeFileSync } from 'node:fs';

const nb = JSON.parse(readFileSync('melanoma_detection_v2.ipynb', 'utf-8'));

function cell(type, source, md) {
  const c = { cell_type: type, metadata: md || {}, source: Array.isArray(source) ? source : [source] };
  if (type === 'code') { c.outputs = []; c.execution_count = null; }
  return c;
}

function copyCell(i) {
  return JSON.parse(JSON.stringify(nb.cells[i]));
}

// ──────────────────────────────────────────────
//  HELPER: data pipeline cells (shared preamble)
// ──────────────────────────────────────────────
function dataPipelineCells() {
  return [
    cell('code', ['!pip install tensorflow --quiet']),
    cell('code', [
      'import os, random, json\n',
      'import numpy as np\n',
      'import tensorflow as tf\n',
      'import matplotlib.pyplot as plt\n',
      'from datetime import datetime\n',
      'from sklearn.metrics import (\n',
      '    classification_report, roc_curve, auc,\n',
      '    confusion_matrix, ConfusionMatrixDisplay,\n',
      ')\n',
      'from tensorflow.keras import layers, Model\n',
      'from tensorflow.keras.applications import VGG16\n',
      '\n',
      'SEED = 42\n',
      'tf.random.set_seed(SEED)\n',
      'np.random.seed(SEED)\n',
      'random.seed(SEED)\n',
      'os.environ["PYTHONHASHSEED"] = str(SEED)',
    ]),
    cell('code', [
      'from google.colab import drive\n',
      'drive.mount("/content/drive")\n',
      '\n',
      'BASE_DIR  = "/content/drive/MyDrive/melanoma_cancer_dataset"\n',
      'TRAIN_DIR = os.path.join(BASE_DIR, "train")\n',
      'TEST_DIR  = os.path.join(BASE_DIR, "test")\n',
      'CLASS_NAMES = ["benign", "malignant"]\n',
      '\n',
      'MODEL_DIR  = "/content/drive/MyDrive/melanoma_model"\n',
      'os.makedirs(MODEL_DIR, exist_ok=True)\n',
      'print("Dataset en:", BASE_DIR)\n',
      'print("Modelos en:", MODEL_DIR)',
    ]),
    copyCell(4),  // markdown: pipeline explanation
    copyCell(5),  // code: AUTOTUNE + dataset loading
    copyCell(6),  // code: data augmentation
    copyCell(7),  // code: visualization
  ];
}

// ──────────────────────────────────────────────
//  HELPER: evaluation + metrics + conversion
// ──────────────────────────────────────────────
function evalAndConvertCells() {
  return [
    cell('markdown', ['## 5. Evaluación sobre el conjunto de test']),
    cell('code', [
      'test_loss, test_acc = modelo.evaluate(test_ds, verbose=1)\n',
      'print(f"\\nTest accuracy : {test_acc:.4f}")\n',
      'print(f"Test loss     : {test_loss:.4f}")\n',
      '\n',
      'y_true_all = []; y_pred_all = []; y_prob_all = []\n',
      'for imgs, labels in test_ds:\n',
      '    preds = modelo.predict(imgs, verbose=0).flatten()\n',
      '    y_true_all.extend(labels.numpy())\n',
      '    y_prob_all.extend(preds)\n',
      '    y_pred_all.extend((preds >= 0.5).astype(int))\n',
      '\n',
      'y_true_all = np.array(y_true_all)\n',
      'y_pred_all = np.array(y_pred_all)\n',
      'y_prob_all = np.array(y_prob_all)\n',
      '\n',
      'print("\\n" + "=" * 60)\n',
      'print("REPORTE DE CLASIFICACIÓN")\n',
      'print("=" * 60)\n',
      'print(classification_report(y_true_all, y_pred_all, target_names=CLASS_NAMES))\n',
      '\n',
      'fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)\n',
      'roc_auc = auc(fpr, tpr)\n',
      'print(f"\\nAUC: {roc_auc:.4f}")\n',
      '\n',
      'cm = confusion_matrix(y_true_all, y_pred_all)\n',
      'tn, fp, fn, tp = cm.ravel()\n',
      'print(f"\\nMatriz de confusión:")\n',
      'print(f"  TN={tn}  FP={fp}")\n',
      'print(f"  FN={fn}  TP={tp}")\n',
      'sens = tp / (tp + fn) if (tp + fn) > 0 else 0\n',
      'spec = tn / (tn + fp) if (tn + fp) > 0 else 0\n',
      'print(f"\\nSensibilidad (Recall maligno): {sens:.4f}")\n',
      'print(f"Especificidad:                 {spec:.4f}")',
    ]),
    copyCell(19),  // ROC + confusion matrix plot

    // Temperature Scaling
    cell('markdown', ['### 5.1 Calibración (Temperature Scaling)']),
    cell('code', [
      'from scipy.optimize import minimize_scalar\n',
      '\n',
      'val_logits = []; val_labels = []\n',
      'for imgs, labels in val_ds:\n',
      '    logits = modelo.predict(imgs, verbose=0).flatten()\n',
      '    logits = np.clip(logits, 1e-7, 1 - 1e-7)\n',
      '    logits = np.log(logits / (1 - logits))\n',
      '    val_logits.extend(logits)\n',
      '    val_labels.extend(labels.numpy())\n',
      'val_logits = np.array(val_logits)\n',
      'val_labels = np.array(val_labels)\n',
      '\n',
      'def nll(t):\n',
      '    cal = 1 / (1 + np.exp(-val_logits / t))\n',
      '    return -np.mean(val_labels * np.log(cal + 1e-7) + (1 - val_labels) * np.log(1 - cal + 1e-7))\n',
      '\n',
      'res = minimize_scalar(nll, bounds=(0.05, 10), method="bounded")\n',
      'T_opt = res.x\n',
      'print(f"Temperatura óptima: {T_opt:.4f}")\n',
      'print(f"Copia en constants.js -> temperature: {T_opt:.3f}")',
    ]),

    // Save model
    cell('markdown', ['## 6. Guardar modelo y convertir a TF.js']),
    cell('code', [
      'ruta_keras = os.path.join(MODEL_DIR, MODELO_ID, f"melanoma_{MODELO_ID}_final.keras")\n',
      'os.makedirs(os.path.dirname(ruta_keras), exist_ok=True)\n',
      'modelo.save(ruta_keras)\n',
      'print(f"Keras model saved: {ruta_keras}")',
    ]),
    cell('code', [
      '# --- Conversion a TF.js con uint8 quantization ---\n',
      '!pip install tensorflowjs --quiet\n',
      'import tensorflowjs as tfjs\n',
      '\n',
      '# TEMPERATURE = T_opt  # descomenta si ya ejecutaste la celda de calibracion\n',
      'TEMPERATURE = 1.0\n',
      '\n',
      'model = tf.keras.models.load_model(ruta_keras)\n',
      'model.summary()\n',
      '\n',
      'converted_dir = os.path.join(MODEL_DIR, MODELO_ID, "tfjs")\n',
      'os.makedirs(converted_dir, exist_ok=True)\n',
      '\n',
      'tfjs.converters.save_keras_model(\n',
      '    model,\n',
      '    converted_dir,\n',
      '    quantization_dtype={"uint8": "*"},\n',
      '    metadata={\n',
      '        "temperature": TEMPERATURE,\n',
      '        "version": "1.0.0",\n',
      '        "modelId": MODELO_ID,\n',
      '        "convertedAt": str(datetime.now()),\n',
      '    },\n',
      ')\n',
      'print(f"TF.js model saved to: {converted_dir}")\n',
      'for f in sorted(os.listdir(converted_dir)):\n',
      '    print(f"  {f}")',
    ]),
    cell('markdown', [
      '---\n',
      '## Conversión completada\n',
      '\n',
      'Copia el modelo TF.js al repositorio:\n',
      '\n',
      '```bash\n',
      `cp -r \"models/\${MODELO_ID}/tfjs/*\" ../demo/public/model/\${MODELO_ID}/\n`,
      '```\n',
      '\n',
      'Actualiza `temperature` en `demo/src/lib/constants.js`.',
    ]),
  ];
}

// ================================================================
// NOTEBOOK 1: VGG16
// ================================================================
{
  const vggCells = [];

  // Title + summary
  vggCells.push(cell('markdown', [
    '# Detección de melanomas — VGG16\n',
    '\n',
    '**Clasificación binaria de lesiones dermatoscópicas (benigno / maligno) con VGG16**\n',
    '\n',
    'Adrián Barriuso Pizarro · 2026\n',
    '\n',
    '---\n',
    '\n',
    '**Arquitectura** | Detalle\n',
    '---|---\n',
    'Backbone | VGG16 (ImageNet) + GAP → Dense(256,ReLU) → Dropout(0.5) → Dense(1,Sigmoid)\n',
    'Dataset | Melanoma Skin Cancer (10 000 imágenes, Kaggle, CC0)\n',
    'Split | Train 7 684 / Val 1 921 / Test 1 000 (50/50 balanceado)\n',
    'Seed | 42\n',
    'Parámetros totales / entrenables | 14 846 273 / 131 585\n',
    '\n',
    '**Fase 1** — VGG16 congelado, RMSprop 1e-4, 20 epochs, EarlyStopping patience=7.\n',
    '\n',
    '**Fase 2** — Últimas 4 capas (block5) descongeladas, Adam 1e-5, 30 epochs.\n',
  ]));

  vggCells.push(...dataPipelineCells());

  // Architecture
  vggCells.push(cell('markdown', ['## 2. Arquitectura del modelo']));
  vggCells.push(cell('code', [
    '# include_top=False descarta las capas densas originales de VGG16\n',
    'base = VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")\n',
    'base.trainable = False  # Fase 1: congelado\n',
    '\n',
    'inputs = layers.Input(shape=(224, 224, 3))\n',
    'x = tf.cast(inputs, tf.float32)\n',
    '# El pipeline ya aplica Rescaling(1/255)\n',
    'x = base(x, training=False)\n',
    'x = layers.GlobalAveragePooling2D()(x)\n',
    'x = layers.Dense(256, activation="relu")(x)\n',
    'x = layers.Dropout(0.5)(x)\n',
    'outputs = layers.Dense(1, activation="sigmoid")(x)\n',
    '\n',
    'modelo = Model(inputs, outputs)\n',
    'modelo.summary()',
  ]));

  // Phase 1
  vggCells.push(cell('markdown', ['## 3. Fase 1 — Extracción de características']));
  vggCells.push(copyCell(11)); // crear_callbacks
  vggCells.push(cell('code', [
    'from tensorflow.keras.optimizers import RMSprop\n',
    '\n',
    'MODELO_ID = "vgg16"\n',
    'class_weight = {0: 1.0, 1: 1.3}\n',
    '\n',
    'callbacks_f1 = crear_callbacks(\n',
    '    f"melanoma_{MODELO_ID}_phase1.keras",\n',
    '    paciencia_es=7, paciencia_lr=3\n',
    ')\n',
    '\n',
    'modelo.compile(\n',
    '    optimizer=RMSprop(learning_rate=1e-4),\n',
    '    loss="binary_crossentropy",\n',
    '    metrics=["accuracy"],\n',
    ')\n',
    '\n',
    'history_f1 = modelo.fit(\n',
    '    train_ds,\n',
    '    validation_data=val_ds,\n',
    '    epochs=20,\n',
    '    class_weight=class_weight,\n',
    '    callbacks=callbacks_f1,\n',
    '    verbose=1,\n',
    ')\n',
    'print(f"\\nMejor val_loss: {min(history_f1.history[\'val_loss\']):.4f}")',
  ]));
  vggCells.push(copyCell(13)); // graficar_historial

  // Phase 2
  vggCells.push(cell('markdown', [
    '## 4. Fase 2 — Fine-tuning\n',
    '\n',
    'Descongelamos las últimas 4 capas de VGG16 (block5_conv1-3, block5_pool).\n',
  ]));
  vggCells.push(cell('code', [
    '# FASE 2: Fine-tuning\n',
    'base.trainable = True\n',
    'for layer in base.layers[:-4]:\n',
    '    layer.trainable = False\n',
    '\n',
    'print("Capas entrenables:")\n',
    'for i, layer in enumerate(base.layers):\n',
    '    if layer.trainable:\n',
    '        print(f"  {i}: {layer.name}")',
  ]));
  vggCells.push(cell('code', [
    'from tensorflow.keras.optimizers import Adam\n',
    '\n',
    'modelo.compile(\n',
    '    optimizer=Adam(learning_rate=1e-5),\n',
    '    loss="binary_crossentropy",\n',
    '    metrics=["accuracy"],\n',
    ')\n',
    '\n',
    'callbacks_f2 = crear_callbacks(\n',
    '    f"melanoma_{MODELO_ID}_finetuning.keras",\n',
    '    paciencia_es=10, paciencia_lr=4\n',
    ')\n',
    '\n',
    'history_f2 = modelo.fit(\n',
    '    train_ds,\n',
    '    validation_data=val_ds,\n',
    '    epochs=30,\n',
    '    class_weight=class_weight,\n',
    '    callbacks=callbacks_f2,\n',
    '    verbose=1,\n',
    ')\n',
    'print(f"\\nMejor val_loss: {min(history_f2.history[\'val_loss\']):.4f}")',
  ]));
  vggCells.push(cell('code', ['graficar_historial(history_f2, "VGG16 — Fine-tuning")']));

  // Evaluation + conversion
  vggCells.push(...evalAndConvertCells());

  const vgg = JSON.parse(JSON.stringify(nb));
  vgg.cells = vggCells;
  vgg.metadata.title = 'VGG16 — Melanoma Detection';
  writeFileSync('notebooks/vgg16.ipynb', JSON.stringify(vgg, null, 1));
  console.log(`Wrote notebooks/vgg16.ipynb (${vggCells.length} cells)`);
}

// ================================================================
// NOTEBOOK 2: ResNet50V2
// ================================================================
{
  const cells = [];

  cells.push(cell('markdown', [
    '# Detección de melanomas — ResNet50V2\n',
    '\n',
    '**Clasificación binaria de lesiones dermatoscópicas (benigno / maligno) con ResNet50V2**\n',
    '\n',
    'Adrián Barriuso Pizarro · 2026\n',
    '\n',
    '---\n',
    '\n',
    '**Protocolo** | Detalle\n',
    '---|---\n',
    'Backbone | ResNet50V2 (ImageNet) + GAP → Dense(256,ReLU) → Dropout(0.5) → Dense(1,Sigmoid)\n',
    'Dataset | Melanoma Skin Cancer (10 000 imágenes, Kaggle, CC0)\n',
    'Split | Train 7 684 / Val 1 921 / Test 1 000 (50/50 balanceado)\n',
    'Seed | 42\n',
    '\n',
    '**Fase 1** — Backbone congelado, RMSprop 1e-4, 20 epochs.\n',
    '\n',
    '**Fase 2** — ~50% capas descongeladas (últimas 80), **BatchNorm congelado**,\n',
    'Adam 1e-6 (más conservador que VGG16 porque ResNet tiene más capacidad).\n',
    'EarlyStopping patience 12.\n',
  ]));

  cells.push(...dataPipelineCells());

  // Architecture
  cells.push(cell('markdown', [
    '## 2. Arquitectura — ResNet50V2\n',
    '\n',
    'ResNet espera entrada en [-1, 1]. Añadimos `Rescaling(2.0, offset=-1.0)`\n',
    'porque la tubería tf.data entrega [0, 1].\n',
    '\n',
    '**Crítico:** `base(x, training=False)` durante Fase 1 para que BatchNorm\n',
    'use sus estadísticas de ImageNet en vez de computar las del batch.\n',
  ]));
  cells.push(cell('code', [
    'from tensorflow.keras.applications import ResNet50V2\n',
    '\n',
    'MODELO_ID = "resnet50v2"\n',
    '\n',
    'base = ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")\n',
    'base.trainable = False\n',
    '\n',
    'inputs = layers.Input(shape=(224, 224, 3))\n',
    'x = layers.Rescaling(2.0, offset=-1.0)(inputs)  # [0,1] -> [-1,1]\n',
    'x = base(x, training=False)\n',
    'x = layers.GlobalAveragePooling2D()(x)\n',
    'x = layers.Dense(256, activation="relu")(x)\n',
    'x = layers.Dropout(0.5)(x)\n',
    'outputs = layers.Dense(1, activation="sigmoid")(x)\n',
    '\n',
    'modelo = Model(inputs, outputs)\n',
    'modelo.summary()',
  ]));

  // Phase 1
  cells.push(cell('markdown', ['## 3. Fase 1 — Extracción de características']));
  cells.push(copyCell(11)); // crear_callbacks
  cells.push(cell('code', [
    'from tensorflow.keras.optimizers import RMSprop\n',
    '\n',
    'class_weight = {0: 1.0, 1: 1.3}\n',
    '\n',
    'callbacks_f1 = crear_callbacks(\n',
    '    f"melanoma_{MODELO_ID}_phase1.keras",\n',
    '    paciencia_es=7, paciencia_lr=3\n',
    ')\n',
    '\n',
    'modelo.compile(\n',
    '    optimizer=RMSprop(learning_rate=1e-4),\n',
    '    loss="binary_crossentropy",\n',
    '    metrics=["accuracy"],\n',
    ')\n',
    '\n',
    'history_f1 = modelo.fit(\n',
    '    train_ds,\n',
    '    validation_data=val_ds,\n',
    '    epochs=20,\n',
    '    class_weight=class_weight,\n',
    '    callbacks=callbacks_f1,\n',
    '    verbose=1,\n',
    ')\n',
    'print(f"\\nMejor val_loss: {min(history_f1.history[\'val_loss\']):.4f}")',
  ]));
  cells.push(copyCell(13)); // graficar_historial

  // Phase 2
  cells.push(cell('markdown', [
    '## 4. Fase 2 — Fine-tuning (BatchNorm congelado)\n',
    '\n',
    'Descongelamos **~50% de las capas no-BN**. BatchNorm se mantiene congelado\n',
    'para evitar que sus estadísticas moving_mean/moving_variance se corrompan\n',
    'con los batches pequeños del fine-tuning (problema clásico de ResNet).\n',
    '\n',
    'LR reducido a **1e-6** (vs 1e-5 de VGG16) porque ResNet tiene más capacidad\n',
    'y es más sensible a cambios grandes de peso.\n',
  ]));
  cells.push(cell('code', [
    '# FASE 2 — Fine-tuning\n',
    'base.trainable = True\n',
    '\n',
    '# Congelar TODAS las capas BatchNorm\n',
    'bn_count = 0\n',
    'for layer in base.layers:\n',
    '    if isinstance(layer, layers.BatchNormalization):\n',
    '        layer.trainable = False\n',
    '        bn_count += 1\n',
    '\n',
    '# Descongelar ~50% de las capas no-BN (últimas 80 de ~160)\n',
    'N_DESCONGELAR = 80\n',
    'for layer in base.layers[-N_DESCONGELAR:]:\n',
    '    if not isinstance(layer, layers.BatchNormalization):\n',
    '        layer.trainable = True\n',
    '\n',
    'trainable_no_bn = sum(1 for l in base.layers\n',
    '    if not isinstance(l, layers.BatchNormalization) and l.trainable)\n',
    'trainable_bn = sum(1 for l in base.layers\n',
    '    if isinstance(l, layers.BatchNormalization) and l.trainable)\n',
    'print(f"Capas entrenables no-BN: {trainable_no_bn} / {len(base.layers)}")\n',
    'print(f"Capas BatchNorm entrenables: {trainable_bn} (deben ser 0)")',
  ]));
  cells.push(cell('code', [
    'from tensorflow.keras.optimizers import Adam\n',
    '\n',
    'modelo.compile(\n',
    '    optimizer=Adam(learning_rate=1e-6),\n',
    '    loss="binary_crossentropy",\n',
    '    metrics=["accuracy"],\n',
    ')\n',
    '\n',
    'callbacks_f2 = crear_callbacks(\n',
    '    f"melanoma_{MODELO_ID}_finetuning.keras",\n',
    '    paciencia_es=12, paciencia_lr=5\n',
    ')\n',
    '\n',
    'history_f2 = modelo.fit(\n',
    '    train_ds,\n',
    '    validation_data=val_ds,\n',
    '    epochs=40,\n',
    '    class_weight=class_weight,\n',
    '    callbacks=callbacks_f2,\n',
    '    verbose=1,\n',
    ')\n',
    'print(f"\\nMejor val_loss: {min(history_f2.history[\'val_loss\']):.4f}")',
  ]));
  cells.push(cell('code', ['graficar_historial(history_f2, "ResNet50V2 — Fine-tuning")']));

  // Evaluation + conversion
  cells.push(...evalAndConvertCells());

  const resnet = JSON.parse(JSON.stringify(nb));
  resnet.cells = cells;
  resnet.metadata.title = 'ResNet50V2 — Melanoma Detection';
  writeFileSync('notebooks/resnet50v2.ipynb', JSON.stringify(resnet, null, 1));
  console.log(`Wrote notebooks/resnet50v2.ipynb (${cells.length} cells)`);
}

// ================================================================
// NOTEBOOK 3: EfficientNetV2S
// ================================================================
{
  const cells = [];

  cells.push(cell('markdown', [
    '# Detección de melanomas — EfficientNetV2S\n',
    '\n',
    '**Clasificación binaria de lesiones dermatoscópicas (benigno / maligno) con EfficientNetV2S**\n',
    '\n',
    'Adrián Barriuso Pizarro · 2026\n',
    '\n',
    '---\n',
    '\n',
    '**Protocolo** | Detalle\n',
    '---|---\n',
    'Backbone | EfficientNetV2S (ImageNet) + GAP → Dense(256,ReLU) → Dropout(0.5) → Dense(1,Sigmoid)\n',
    'Dataset | Melanoma Skin Cancer (10 000 imágenes, Kaggle, CC0)\n',
    'Split | Train 7 684 / Val 1 921 / Test 1 000 (50/50 balanceado)\n',
    'Seed | 42\n',
    '\n',
    'EfficientNetV2S ofrece el mejor ratio accuracy/parámetros de su familia.\n',
    'Con ~22 M parámetros y ~15 MB en TF.js cuantizado, es comparable en tamaño\n',
    'a VGG16 pero con mejor precisión teórica.\n',
    '\n',
    '**Fase 1** — Backbone congelado, RMSprop 1e-4, 20 epochs.\n',
    '\n',
    '**Fase 2** — ~50% capas descongeladas, **BatchNorm congelado**, Adam 1e-5.\n',
  ]));

  cells.push(...dataPipelineCells());

  // Architecture
  cells.push(cell('markdown', [
    '## 2. Arquitectura — EfficientNetV2S\n',
    '\n',
    'Usa `preprocess_input` de EfficientNetV2 (normalización con stats ImageNet).\n',
  ]));
  cells.push(cell('code', [
    'from tensorflow.keras.applications import EfficientNetV2S\n',
    'from tensorflow.keras.applications.efficientnet_v2 import preprocess_input\n',
    '\n',
    'MODELO_ID = "efficientnetv2s"\n',
    '\n',
    'base = EfficientNetV2S(\n',
    '    input_shape=(224, 224, 3),\n',
    '    include_top=False,\n',
    '    weights="imagenet",\n',
    ')\n',
    'base.trainable = False\n',
    '\n',
    'inputs = layers.Input(shape=(224, 224, 3))\n',
    'x = preprocess_input(inputs)\n',
    'x = base(x, training=False)\n',
    'x = layers.GlobalAveragePooling2D()(x)\n',
    'x = layers.Dense(256, activation="relu")(x)\n',
    'x = layers.Dropout(0.5)(x)\n',
    'outputs = layers.Dense(1, activation="sigmoid")(x)\n',
    '\n',
    'modelo = Model(inputs, outputs)\n',
    'modelo.summary()',
  ]));

  // Phase 1
  cells.push(cell('markdown', ['## 3. Fase 1 — Extracción de características']));
  cells.push(copyCell(11)); // crear_callbacks
  cells.push(cell('code', [
    'from tensorflow.keras.optimizers import RMSprop\n',
    '\n',
    'class_weight = {0: 1.0, 1: 1.3}\n',
    '\n',
    'callbacks_f1 = crear_callbacks(\n',
    '    f"melanoma_{MODELO_ID}_phase1.keras",\n',
    '    paciencia_es=7, paciencia_lr=3\n',
    ')\n',
    '\n',
    'modelo.compile(\n',
    '    optimizer=RMSprop(learning_rate=1e-4),\n',
    '    loss="binary_crossentropy",\n',
    '    metrics=["accuracy"],\n',
    ')\n',
    '\n',
    'history_f1 = modelo.fit(\n',
    '    train_ds,\n',
    '    validation_data=val_ds,\n',
    '    epochs=20,\n',
    '    class_weight=class_weight,\n',
    '    callbacks=callbacks_f1,\n',
    '    verbose=1,\n',
    ')\n',
    'print(f"\\nMejor val_loss: {min(history_f1.history[\'val_loss\']):.4f}")',
  ]));
  cells.push(copyCell(13)); // graficar_historial

  // Phase 2
  cells.push(cell('markdown', [
    '## 4. Fase 2 — Fine-tuning\n',
    '\n',
    'Descongelamos ~50% de las capas. BatchNorm congelado (EfficientNetV2 también\n',
    'los usa en sus bloques MBConv).\n',
  ]));
  cells.push(cell('code', [
    '# FASE 2 — Fine-tuning\n',
    'base.trainable = True\n',
    '\n',
    '# Congelar BatchNorm\n',
    'for layer in base.layers:\n',
    '    if isinstance(layer, layers.BatchNormalization):\n',
    '        layer.trainable = False\n',
    '\n',
    '# Descongelar ~50% de las capas restantes\n',
    'N = len(base.layers) // 2\n',
    'for layer in base.layers[-N:]:\n',
    '    if not isinstance(layer, layers.BatchNormalization):\n',
    '        layer.trainable = True\n',
    '\n',
    'trainable_no_bn = sum(1 for l in base.layers\n',
    '    if not isinstance(l, layers.BatchNormalization) and l.trainable)\n',
    'trainable_bn = sum(1 for l in base.layers\n',
    '    if isinstance(l, layers.BatchNormalization) and l.trainable)\n',
    'print(f"Capas entrenables no-BN: {trainable_no_bn} / {len(base.layers)}")\n',
    'print(f"Capas BatchNorm entrenables: {trainable_bn} (deben ser 0)")',
  ]));
  cells.push(cell('code', [
    'from tensorflow.keras.optimizers import Adam\n',
    '\n',
    'modelo.compile(\n',
    '    optimizer=Adam(learning_rate=1e-5),\n',
    '    loss="binary_crossentropy",\n',
    '    metrics=["accuracy"],\n',
    ')\n',
    '\n',
    'callbacks_f2 = crear_callbacks(\n',
    '    f"melanoma_{MODELO_ID}_finetuning.keras",\n',
    '    paciencia_es=12, paciencia_lr=5\n',
    ')\n',
    '\n',
    'history_f2 = modelo.fit(\n',
    '    train_ds,\n',
    '    validation_data=val_ds,\n',
    '    epochs=40,\n',
    '    class_weight=class_weight,\n',
    '    callbacks=callbacks_f2,\n',
    '    verbose=1,\n',
    ')\n',
    'print(f"\\nMejor val_loss: {min(history_f2.history[\'val_loss\']):.4f}")',
  ]));
  cells.push(cell('code', ['graficar_historial(history_f2, "EfficientNetV2S — Fine-tuning")']));

  // Evaluation + conversion
  cells.push(...evalAndConvertCells());

  const eff = JSON.parse(JSON.stringify(nb));
  eff.cells = cells;
  eff.metadata.title = 'EfficientNetV2S — Melanoma Detection';
  writeFileSync('notebooks/efficientnetv2s.ipynb', JSON.stringify(eff, null, 1));
  console.log(`Wrote notebooks/efficientnetv2s.ipynb (${cells.length} cells)`);
}

console.log('\nAll notebooks generated.');
