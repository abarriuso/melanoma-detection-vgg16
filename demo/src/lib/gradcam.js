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
      } catch {}
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
  const clsModel = tf.model({ inputs: targetLayer.output, outputs: model.output });

  const result = { actModel, clsModel };
  splitCache.set(model, result);
  return result;
}

function getGradFn(clsModel) {
  if (gradCache.has(clsModel)) return gradCache.get(clsModel);
  const fn = tf.grad((activations) => {
    const pred = clsModel.predict(activations);
    return pred.squeeze();
  });
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
    activations = actModel.predict(input);
    grads = gradFn(activations);
    pooledGrads = grads.mean([0, 1], true);
    cam = tf.tidy(() => {
      const weighted = activations.mul(pooledGrads);
      const summed = weighted.sum(-1).squeeze();
      const relued = summed.maximum(0);
      const norm = relued.max();
      return norm > 0 ? relued.div(norm) : relued;
    });
    const result = await cam.resizeBilinear([224, 224]).data();
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
