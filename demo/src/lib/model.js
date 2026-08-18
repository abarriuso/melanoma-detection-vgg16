import * as tf from '@tensorflow/tfjs';
import { getModel } from './constants';

const BASE = import.meta.env.BASE_URL;
const modelCache = new Map();
const metaCache = new Map();
let activeModelId = 'efficientnetv2s';

function disposeEntry(entry) {
  // La caché guarda la Promise mientras el modelo carga y el modelo
  // resuelto después; hay que disponer ambos casos.
  if (entry instanceof Promise) {
    entry.then((m) => m?.dispose?.()).catch(() => {});
  } else {
    entry?.dispose?.();
  }
}

/**
 * Libera los pesos (GPU/CPU) de un modelo cacheado y lo saca de la caché.
 * @param {string} id
 */
export function disposeModel(id) {
  const entry = modelCache.get(id);
  if (entry) disposeEntry(entry);
  modelCache.delete(id);
  metaCache.delete(id);
}

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
  if (!entry) return Promise.reject(new Error(`Modelo "${id}" no encontrado en el registro`));

  const cached = modelCache.get(id);
  if (cached) {
    /* Si ya está cargando (Promise pendiente), reusarla */
    if (cached instanceof Promise) return cached;
    const meta = metaCache.get(id);
    if (meta?.version !== entry.version) {
      console.warn(`Version mismatch for ${id}, reloading...`);
      // Disponer el modelo viejo: borrar solo la entrada fugaba sus pesos.
      disposeModel(id);
    } else {
      return cached;
    }
  }

  const url = `${BASE}${entry.path}`;
  const promise = ensureBackend().then(async () => {
    const model = await tf.loadLayersModel(url, { onProgress });
    // Warmup: la primera inferencia real de un modelo grande compila los
    // shaders WebGL sobre la marcha, lo que puede tardar varios segundos
    // (más en GPUs débiles). Lo hacemos aquí, mientras el modelo todavía
    // se muestra como "cargando", para que el usuario no lo note al pulsar
    // "Analizar imagen" por primera vez.
    try {
      const warm = tf.tidy(() => model.predict(tf.zeros([1, 224, 224, 3])));
      await warm.data();
      warm.dispose();
    } catch {
      // Si el warmup falla, la inferencia real lo intentará de nuevo
      // (y su propio error se gestiona en predictImage/analizar).
    }
    if (modelCache.get(id) !== promise) {
      // La entrada fue reemplazada o expulsada mientras cargaba: este
      // modelo ya no es el activo, disponerlo para no fugar sus pesos.
      model.dispose?.();
      return model;
    }
    if (model.userDefinedMetadata) {
      metaCache.set(id, model.userDefinedMetadata);
    } else {
      metaCache.set(id, { version: entry.version, temperature: entry.temperature });
    }
    // Guardar el modelo resuelto (no la Promise) para poder disponerlo.
    modelCache.set(id, model);
    // Política de caché: conservar solo el modelo activo. Los tres modelos
    // juntos suman ~60 MB de pesos residentes en GPU, así que al terminar
    // de cargar uno se dispone cualquier otro que hubiera en caché.
    for (const otherId of [...modelCache.keys()]) {
      if (otherId !== id) disposeModel(otherId);
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

// ¿El renderer WebGL es software (sin aceleración por hardware)?
// SwiftShader (Chrome/Edge), llvmpipe (Linux/Mesa) y el Microsoft Basic
// Render Driver son los rasterizadores software habituales: con cualquiera
// de ellos una VGG16 pasa de ~0,2 s a minutos por análisis.
export function isSoftwareRenderer(renderer) {
  if (!renderer) return false;
  return /swiftshader|llvmpipe|softpipe|software|microsoft basic render/i.test(renderer);
}

// Inspecciona el contexto WebGL real del navegador para poder avisar al
// usuario si tiene la aceleración gráfica desactivada. No lanza nunca:
// en entornos sin canvas (tests) devuelve `supported: false`.
export function getGpuInfo() {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return { supported: false, renderer: null, software: false };
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = dbg
      ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER) || '';
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return { supported: true, renderer, software: isSoftwareRenderer(renderer) };
  } catch {
    return { supported: false, renderer: null, software: false };
  }
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
