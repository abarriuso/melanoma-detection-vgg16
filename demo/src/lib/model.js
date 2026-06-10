import * as tf from '@tensorflow/tfjs';
import { getModel } from './constants';

const BASE = import.meta.env.BASE_URL;
const modelCache = new Map();
const metaCache = new Map();
let activeModelId = 'vgg16';

let backendPromise = null;
async function ensureBackend() {
  if (backendPromise) return backendPromise;
  backendPromise = (async () => {
    // Intentar WebGL (GPU). TF.js no tiene webgpu instalado.
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      return;
    } catch {
      // Si falla WebGL (entorno sin GPU), usar CPU
    }
    await tf.setBackend('cpu');
    await tf.ready();
  })();
  return backendPromise;
}

export function calibrate(p, temperature) {
  if (temperature == null || temperature === 1.0) return p;
  const eps = 1e-7;
  const clamped = Math.min(Math.max(p, eps), 1 - eps);
  const logit = Math.log(clamped / (1 - clamped));
  return 1 / (1 + Math.exp(-logit / temperature));
}

/**
 * Carga un modelo específico por ID. Usa caché por modelo.
 * @param {string} modelId
 * @param {(fraction: number) => void} [onProgress]
 * @returns {Promise<tf.LayersModel>}
 */
export async function loadModel(modelId, onProgress) {
  const id = modelId || activeModelId;
  const entry = getModel(id);

  const cached = modelCache.get(id);
  if (cached) {
    const meta = metaCache.get(id);
    if (meta?.version !== entry.version) {
      console.warn(`Version mismatch for ${id}, reloading...`);
      modelCache.delete(id);
      metaCache.delete(id);
    } else {
      return cached;
    }
  }

  const url = `${BASE}${entry.path}`;
  const promise = ensureBackend().then(async () => {
    const model = await tf.loadLayersModel(url, { onProgress });
    if (model.userDefinedMetadata) {
      metaCache.set(id, model.userDefinedMetadata);
    } else {
      metaCache.set(id, { version: entry.version, temperature: entry.temperature });
    }
    return model;
  }, (err) => {
    modelCache.delete(id);
    metaCache.delete(id);
    throw err;
  });

  modelCache.set(id, promise);
  return promise;
}

export function getActiveModelId() {
  return activeModelId;
}

export function setActiveModelId(id) {
  activeModelId = id;
}

export function getModelMetadata(modelId) {
  const id = modelId || activeModelId;
  return metaCache.get(id) || getModel(id);
}

export function getBackend() {
  return tf.getBackend();
}

/**
 * Clasifica una imagen.
 * @param {HTMLImageElement} imgElement
 * @param {string} [modelId] - opcional, usa activo si no se pasa
 * @returns {Promise<{raw: number, calibrated: number}>}
 */
export async function predictImage(imgElement, modelId) {
  const id = modelId || activeModelId;
  const model = await loadModel(id);
  const meta = metaCache.get(id) || getModel(id);
  const temperature = meta.temperature ?? null;

  const pixels = await tf.browser.fromPixelsAsync(imgElement);
  let input;
  let output;
  try {
    input = tf.tidy(() =>
      pixels
        .resizeBilinear([224, 224])
        .toFloat()
        .div(255)
        .expandDims(0)
    );
    output = model.predict(input);
    const raw = (await output.data())[0];
    const safe = Math.min(Math.max(raw, 1e-7), 1 - 1e-7);
    return { raw: safe, calibrated: calibrate(safe, temperature) };
  } finally {
    pixels.dispose();
    input?.dispose();
    output?.dispose();
  }
}
