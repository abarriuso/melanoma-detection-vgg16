import * as tf from '@tensorflow/tfjs';
import { getModel } from './constants';

const TARGET_LAYER_CANDIDATES = ['block5_conv3', 'block5_conv3_1', 'conv5_block3_conv'];
const splitCache = new WeakMap();
const gradCache = new WeakMap();

function findTargetLayer(model, modelId) {
  if (modelId) {
    const entry = getModel(modelId);
    if (entry?.targetLayer) {
      try {
        const named = model.getLayer(entry.targetLayer);
        if (named) return named;
      } catch {
        // capa no encontrada por nombre; caemos a los candidatos genéricos
      }
    }
  }
  for (const name of TARGET_LAYER_CANDIDATES) {
    const layer = model.getLayer(name);
    if (layer) return layer;
  }
  const convLayers = model.layers.filter(
    (l) => l.outputShape && l.outputShape.length === 4 && l.getClassName().includes('Conv'),
  );
  return convLayers[convLayers.length - 1];
}

function getSplitModels(model, modelId) {
  if (splitCache.has(model)) return splitCache.get(model);

  const targetLayer = findTargetLayer(model, modelId);
  const actModel = tf.model({ inputs: model.inputs, outputs: targetLayer.output });

  // tf.model() exige que `inputs` sea un InputLayer: no se puede cortar el
  // grafo por un tensor intermedio (lanza "Input layers to a LayersModel
  // must be InputLayer objects"). El clasificador se reconstruye aplicando
  // las capas posteriores a la capa objetivo sobre una entrada nueva con la
  // forma de sus activaciones. Vale porque la cabeza tras block5_conv3 es
  // una cadena lineal (pool → GAP → Dense → Dropout → Dense).
  const idx = model.layers.indexOf(targetLayer);
  const clsInput = tf.input({ shape: targetLayer.outputShape.slice(1) });
  let y = clsInput;
  for (let i = idx + 1; i < model.layers.length; i++) {
    y = model.layers[i].apply(y);
  }
  const clsModel = tf.model({ inputs: clsInput, outputs: y });

  const result = { actModel, clsModel };
  splitCache.set(model, result);
  return result;
}

function getGradFn(clsModel) {
  if (gradCache.has(clsModel)) return gradCache.get(clsModel);
  // apply() en vez de predict(): predict corre fuera de la cinta de
  // gradientes y tf.grad no podría derivar a través de él.
  const fn = tf.grad((activations) => clsModel.apply(activations).squeeze());
  gradCache.set(clsModel, fn);
  return fn;
}

export async function computeGradCAM(model, imgElement, modelId) {
  const { actModel, clsModel } = getSplitModels(model, modelId);
  const gradFn = getGradFn(clsModel);

  const pixels = await tf.browser.fromPixelsAsync(imgElement);
  let input;
  let activations;
  let grads;
  let pooledGrads;
  let cam;
  try {
    input = tf.tidy(() =>
      pixels
        .resizeBilinear([224, 224])
        .toFloat()
        .div(255)
        .expandDims(0)
    );
    activations = actModel.predict(input);          // [1, h, w, c]
    grads = gradFn(activations);                    // [1, h, w, c]
    // Peso de cada canal: promedio de su gradiente sobre el plano espacial
    // (ejes 1 y 2 = alto y ancho). Promediar [0, 1] (batch y alto) produce
    // un peso distinto por columna, que el broadcasting acepta en silencio
    // pero no es Grad-CAM.
    pooledGrads = grads.mean([1, 2], true);         // [1, 1, 1, c]
    cam = tf.tidy(() => {
      const weighted = activations.mul(pooledGrads);
      const summed = weighted.sum(-1).squeeze([0]); // [h, w]
      const relued = summed.maximum(0);
      // max() devuelve un tensor escalar, no un número: compararlo con
      // `> 0` en JS es siempre false. maximum(eps) evita dividir por cero
      // sin salir del grafo.
      const norm = relued.max().maximum(1e-8);
      // resizeBilinear exige rango 3 o 4; con [h, w] a secas lanza.
      return relued.div(norm).expandDims(-1);       // [h, w, 1]
    });
    const resized = cam.resizeBilinear([224, 224]); // [224, 224, 1]
    const result = await resized.data();
    resized.dispose();
    return Array.from(result);
  } finally {
    pixels.dispose();
    input?.dispose();
    activations?.dispose();
    grads?.dispose();
    pooledGrads?.dispose();
    cam?.dispose();
  }
}

const COLORMAP = [
  [0.0, 0.0, 0.0],
  [0.1, 0.0, 0.2],
  [0.2, 0.0, 0.5],
  [0.3, 0.1, 0.7],
  [0.4, 0.2, 0.9],
  [0.5, 0.4, 0.8],
  [0.6, 0.6, 0.6],
  [0.7, 0.8, 0.4],
  [0.8, 0.9, 0.2],
  [0.9, 0.95, 0.05],
  [1.0, 1.0, 0.0],
];

function jet(t) {
  const idx = t * (COLORMAP.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, COLORMAP.length - 1);
  const f = idx - lo;
  return [
    COLORMAP[lo][0] + (COLORMAP[hi][0] - COLORMAP[lo][0]) * f,
    COLORMAP[lo][1] + (COLORMAP[hi][1] - COLORMAP[lo][1]) * f,
    COLORMAP[lo][2] + (COLORMAP[hi][2] - COLORMAP[lo][2]) * f,
  ];
}

export function paintHeatmap(canvas, heatmap, width, height) {
  if (heatmap == null) return;
  if (width) canvas.width = width;
  if (height) canvas.height = height;
  const ctx = canvas.getContext('2d');
  const w = width || canvas.width;
  const h = height || canvas.height;
  const imgData = ctx.createImageData(w, h);
  const d = imgData.data;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const t = Math.min(Math.max(heatmap[idx], 0), 1);
      const [r, g, b] = jet(t);
      const alpha = t * 0.65;
      const pi = idx * 4;
      d[pi] = r * 255;
      d[pi + 1] = g * 255;
      d[pi + 2] = b * 255;
      d[pi + 3] = alpha * 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
}
