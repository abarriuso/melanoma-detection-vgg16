#!/usr/bin/env node
/**
 * Re-evalúa un modelo TF.js de la demo sobre el conjunto de test LIMPIO
 * (sin las imágenes con duplicado o cuasi-duplicado en train, según
 * `internal/dedup_report.json`, generado por `scripts/dedup_test.py`).
 *
 * Contexto: la auditoría 02 detectó que ~23-27 % del test tiene
 * cuasi-duplicados en train (aHash 16x16, Hamming <= 12), así que las
 * métricas publicadas sobre las 1 000 imágenes son una cota optimista.
 * Este script mide el modelo TF.js servido (float32) sobre el test limpio.
 *
 * Preprocesado: idéntico al de la demo (demo/src/lib/model.js):
 *   decodificar JPEG -> resizeBilinear(224,224) -> /255 -> modelo.
 * La decodificación JPEG la hace `jpeg-js` (en el navegador la hace el
 * decodificador del propio navegador); puede haber diferencias de sub-LSB
 * respecto al navegador, despreciables para las métricas.
 *
 * Calibración: sigmoide(logit(p) / T) con la T de constants.js, ajustada
 * sobre validación en Kaggle. Umbral de decisión 0.5 (como la demo).
 *
 * Uso (desde la raíz del repo):
 *   node scripts/reval_tfjs.mjs --model efficientnetv2s [--budget 540] [--limit N]
 *
 * Salida: JSON por stdout (y, si se pasa --out, también a fichero).
 */

import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
// Los paquetes (@tensorflow/tfjs, jpeg-js) viven en internal/reval (scratch).
const require = createRequire(join(ROOT, 'internal', 'reval', 'package.json'));
const tf = require('@tensorflow/tfjs');
const jpeg = require('jpeg-js');
try {
  const wasm = require('@tensorflow/tfjs-backend-wasm');
  wasm.setWasmPaths(
    join(ROOT, 'internal', 'reval', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist') + '/'
  );
} catch {
  /* backend wasm no disponible: se usará cpu */
}

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}

const MODEL_ID = arg('model', 'efficientnetv2s');
const BUDGET_S = Number(arg('budget', 540)); // presupuesto de cómputo (~9 min)
const LIMIT = Number(arg('limit', 0)); // 0 = sin límite
const OUT = arg('out', null);
const BACKEND = arg('backend', 'wasm');
const FULL = arg('full', '0') === '1'; // 1 = test completo (chequeo de sanidad)
const DECODE = arg('decode', 'ppm'); // 'ppm' (libjpeg-turbo, = Kaggle) | 'jpeg' (jpeg-js)
const OFFSET = Number(arg('offset', '0')); // índice dentro de la selección para reanudar
const DUMP = arg('dump', null); // fichero JSONL con probs por imagen
const SPLIT = arg('split', 'test'); // 'test' | 'train' (train = chequeo de memorización)

const META = {
  efficientnetv2s: { temperature: 1.1836391169830454 },
  resnet50v2: { temperature: 1.0220809323332534 },
  vgg16: { temperature: 1.3358887532544077 },
};

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function calibrate(p, T) {
  const s = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
  return sigmoid(Math.log(s / (1 - s)) / T);
}

// IC95 de Wilson para una proporción x/n
function wilson(x, n, z = 1.959964) {
  if (n === 0) return [NaN, NaN];
  const p = x / n;
  const den = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / den;
  const half = (z / den) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [center - half, center + half];
}

// AUC por Mann-Whitney (ranking medio para empates)
function auc(scoresPos, scoresNeg) {
  const all = [
    ...scoresPos.map((s) => ({ s, pos: true })),
    ...scoresNeg.map((s) => ({ s, pos: false })),
  ].sort((a, b) => a.s - b.s);
  let rankSum = 0;
  let i = 0;
  while (i < all.length) {
    let j = i;
    while (j + 1 < all.length && all[j + 1].s === all[i].s) j++;
    const rankMedio = (i + 1 + j + 1) / 2;
    for (let k = i; k <= j; k++) if (all[k].pos) rankSum += rankMedio;
    i = j + 1;
  }
  const np = scoresPos.length;
  const nn = scoresNeg.length;
  return (rankSum - (np * (np + 1)) / 2) / (np * nn);
}

function decodeToTensor(buf) {
  const { width, height, data } = jpeg.decode(buf, {
    maxMemoryUsageInMB: 512,
    formatAsRGBA: true,
  });
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    rgb[i] = data[j];
    rgb[i + 1] = data[j + 1];
    rgb[i + 2] = data[j + 2];
  }
  return tf.tensor3d(rgb, [height, width, 3], 'int32');
}

// Decodifica un PPM P6 (RGB crudo, generado por scripts/decode_test_ppm.py
// con Pillow/libjpeg-turbo, el mismo decodificador que TF en Kaggle).
function ppmToTensor(buf) {
  // Cabecera: "P6\n<w> <h>\n255\n" (Pillow no escribe comentarios)
  let off = 0;
  const token = () => {
    while (buf[off] <= 32) off++;
    let s = off;
    while (buf[off] > 32) off++;
    return buf.subarray(s, off).toString('ascii');
  };
  const magic = token();
  if (magic !== 'P6') throw new Error(`PPM inesperado: ${magic}`);
  const width = Number(token());
  const height = Number(token());
  const maxval = Number(token());
  if (maxval !== 255) throw new Error(`maxval inesperado: ${maxval}`);
  off++; // un solo byte de whitespace tras maxval
  return tf.tensor3d(
    new Uint8Array(buf.buffer, buf.byteOffset + off, width * height * 3),
    [height, width, 3],
    'int32'
  );
}

async function main() {
  const t0 = Date.now();
  const report = JSON.parse(
    readFileSync(join(ROOT, 'internal', 'dedup_report.json'), 'utf8')
  );
  const contaminadas = new Set(report.test_contaminadas_umbral_principal);

  const datasetDir = join(ROOT, 'archive', 'melanoma_cancer_dataset', SPLIT);
  // Intercalar clases (determinista) para que un corte por presupuesto de
  // tiempo deje una muestra estratificada 50/50 y no solo benignos.
  const items = [];
  const porClase = [[], []];
  for (const [clase, label] of [['benign', 0], ['malignant', 1]]) {
    for (const f of readdirSync(join(datasetDir, clase)).sort()) {
      const rel = `archive/melanoma_cancer_dataset/${SPLIT}/${clase}/${f}`;
      if (SPLIT === 'test' && !FULL && contaminadas.has(rel)) continue;
      const src = join(datasetDir, clase, f);
      const path =
        DECODE === 'ppm'
          ? join(ROOT, 'internal', 'reval', 'ppm', SPLIT, clase, f.replace(/\.[^.]+$/, '.ppm'))
          : src;
      porClase[label].push({ path, rel, label });
    }
  }
  for (let i = 0; i < Math.max(porClase[0].length, porClase[1].length); i++) {
    if (i < porClase[0].length) items.push(porClase[0][i]);
    if (i < porClase[1].length) items.push(porClase[1][i]);
  }
  let seleccion = items;
  if (OFFSET > 0) seleccion = seleccion.slice(OFFSET);
  if (LIMIT > 0) seleccion = seleccion.slice(0, LIMIT);
  console.error(
    FULL
      ? `MODO SANIDAD: test completo ${items.length} imagenes con ${MODEL_ID}`
      : `Test limpio: ${items.length} imagenes ` +
          `(excluidas ${contaminadas.size} contaminadas de ${items.length + contaminadas.size}); ` +
          `evaluando ${seleccion.length} con ${MODEL_ID}`
  );

  const okBackend = await tf.setBackend(BACKEND);
  console.error(`Backend tfjs: ${tf.getBackend()} (solicitado ${BACKEND}, ok=${okBackend})`);

  // Node 24 no soporta file:// en fetch: cargamos el modelo desde memoria.
  const modelDir = join(ROOT, 'demo', 'public', 'model', MODEL_ID);
  const modelJson = JSON.parse(readFileSync(join(modelDir, 'model.json'), 'utf8'));
  const weightSpecs = [];
  const shards = [];
  for (const group of modelJson.weightsManifest) {
    weightSpecs.push(...group.weights);
    for (const p of group.paths) {
      shards.push(readFileSync(join(modelDir, p)));
    }
  }
  const weightData = Buffer.concat(shards).buffer;
  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      format: modelJson.format,
      generatedBy: modelJson.generatedBy,
      convertedBy: modelJson.convertedBy,
      userDefinedMetadata: modelJson.userDefinedMetadata,
      weightSpecs,
      weightData,
    })
  );
  const T = META[MODEL_ID].temperature;

  const results = [];
  let cortadoPorTiempo = false;
  const BATCH = Number(arg('batch', '8'));
  for (let base = 0; base < seleccion.length; base += BATCH) {
    if (Date.now() - t0 > BUDGET_S * 1000) {
      cortadoPorTiempo = true;
      break;
    }
    const lote = seleccion.slice(base, base + BATCH);
    const decodificar = DECODE === 'ppm' ? ppmToTensor : decodeToTensor;
    const probs = tf.tidy(() => {
      const tensores = lote.map((it) =>
        decodificar(readFileSync(it.path))
          .resizeBilinear([224, 224])
          .toFloat()
          .div(255)
      );
      const input = tf.stack(tensores);
      const out = model.predict(input);
      return Array.from(out.dataSync());
    });
    lote.forEach((it, k) =>
      results.push({ rel: it.rel, label: it.label, raw: probs[k], cal: calibrate(probs[k], T) })
    );
    if (results.length % 50 < BATCH) {
      console.error(
        `  ${results.length}/${seleccion.length} (${((Date.now() - t0) / 1000).toFixed(0)} s)`
      );
    }
  }

  const n = results.length;
  let tp = 0, tn = 0, fp = 0, fn = 0;
  const pos = [], neg = [];
  for (const r of results) {
    const pred = r.cal >= 0.5 ? 1 : 0;
    if (r.label === 1) pos.push(r.raw); else neg.push(r.raw);
    if (pred === 1 && r.label === 1) tp++;
    else if (pred === 1) fp++;
    else if (r.label === 1) fn++;
    else tn++;
  }
  const acc = (tp + tn) / n;
  const sens = pos.length ? tp / pos.length : NaN;
  const spec = neg.length ? tn / neg.length : NaN;

  const salida = {
    modelo: MODEL_ID,
    fuente_modelo: `demo/public/model/${MODEL_ID}/model.json (TF.js float32, el servido en la demo)`,
    backend: `tfjs (${tf.getBackend()}, Node ${process.version})`,
    nota_backend:
      DECODE === 'ppm'
        ? 'Decodificación JPEG con Pillow/libjpeg-turbo (scripts/decode_test_ppm.py), ' +
          'el mismo decodificador que TF en Kaggle; resize bilinear 224x224 en tfjs ' +
          '(idéntico a la demo y al entrenamiento).'
        : 'Decodificación JPEG con jpeg-js (desviación conocida respecto a libjpeg).',
    test_total: items.length + contaminadas.size,
    test_excluidas_contaminadas: contaminadas.size,
    test_limpio_disponible: items.length,
    n_evaluadas: n,
    cortado_por_presupuesto_tiempo: cortadoPorTiempo,
    presupuesto_s: BUDGET_S,
    umbral: 0.5,
    temperature: T,
    n_pos: pos.length,
    n_neg: neg.length,
    tp, tn, fp, fn,
    accuracy: acc,
    sensibilidad: sens,
    especificidad: spec,
    auc,
    ic95_wilson: {
      accuracy: wilson(tp + tn, n),
      sensibilidad: wilson(tp, pos.length),
      especificidad: wilson(tn, neg.length),
    },
    duracion_s: Math.round((Date.now() - t0) / 1000),
  };
  const json = JSON.stringify(salida, null, 2);
  if (OUT) writeFileSync(OUT, json, 'utf8');
  if (DUMP) {
    writeFileSync(
      DUMP,
      results.map((r) => JSON.stringify({ rel: r.rel, label: r.label, raw: r.raw })).join('\n') + '\n',
      'utf8'
    );
  }
  console.log(json);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
