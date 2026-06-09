import * as tf from '@tensorflow/tfjs';

// BASE_URL incluye el path del repo (p.ej. /melanoma-detection-vgg16/) para
// que las rutas funcionen correctamente en GitHub Pages, donde la app no está
// en la raíz del dominio sino en un subdirectorio.
const MODEL_URL = `${import.meta.env.BASE_URL}model/model.json`;

// Versión del modelo para invalidar caché cuando se actualiza.
// Debe coincidir con la versión en model.json (userDefinedMetadata.version).
const MODEL_VERSION = '1.0.0';

// Patrón singleton: el modelo pesa ~15 MB (tras cuantización uint8) y tarda
// varios segundos en cargar. Guardamos la promesa para no descargarlo de nuevo.
let modelPromise = null;
let modelMetadata = null;

// Inicialización del backend. WebGPU (2-3× más rápido que WebGL en hardware
// compatible) se carga con dynamic import para que el código solo viaje al
// cliente cuando el navegador realmente lo soporta. Si falla por cualquier
// motivo, TF.js cae al backend WebGL/CPU por defecto sin romper la app.
let backendPromise = null;
function ensureBackend() {
  if (backendPromise) return backendPromise;
  backendPromise = (async () => {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        await import('@tensorflow/tfjs-backend-webgpu');
        await tf.setBackend('webgpu');
        await tf.ready();
        return;
      } catch (err) {
        console.warn('WebGPU no disponible, usando backend por defecto:', err);
      }
    }
    await tf.ready(); // garantiza que webgl/cpu estén inicializados
  })();
  return backendPromise;
}

// Temperature scaling: corrige la sobreconfianza del modelo para que las
// probabilidades mostradas reflejen la realidad. El valor lo calcula el
// notebook (sección 8.3, "Calibración") y se imprime al final de esa celda.
// T = 1.0 equivale a no calibrar.
// T = 0.902 reduce el ECE de 0.031 a 0.025 sobre el conjunto de test.
// Valor por defecto; se sobrescribe con model.json metadata si está disponible.
let TEMPERATURE = 0.902;

// Aplica temperature scaling a una probabilidad sigmoid.
// No altera la clasificación (el umbral 0.5 es invariante), solo la confianza.
export function calibrate(p) {
  if (TEMPERATURE === 1.0) return p;
  const eps = 1e-7;
  const clamped = Math.min(Math.max(p, eps), 1 - eps);
  const logit = Math.log(clamped / (1 - clamped));
  return 1 / (1 + Math.exp(-logit / TEMPERATURE));
}

/**
 * Carga el modelo TF.js (singleton) con verificación de versión.
 * @param {(fraction: number) => void} [onProgress] callback 0..1 de descarga
 * @returns {Promise<import('@tensorflow/tfjs').LayersModel>}
 */
export async function loadModel(onProgress) {
  if (modelPromise) {
    // Verificar versión en metadata si ya tenemos el modelo cargado
    if (modelMetadata !== null && modelMetadata?.version !== MODEL_VERSION) {
      console.warn(`Model version mismatch (cached: ${modelMetadata?.version}, expected: ${MODEL_VERSION}). Reloading...`);
      modelPromise = null;
      modelMetadata = null;
    } else {
      return modelPromise;
    }
  }

  modelPromise = ensureBackend().then(async () => {
    const model = await tf.loadLayersModel(MODEL_URL, { onProgress });
    
    // Extraer metadata del modelo (versión, temperature, etc.)
    if (model.userDefinedMetadata) {
      modelMetadata = model.userDefinedMetadata;
      if (modelMetadata.temperature) {
        TEMPERATURE = modelMetadata.temperature;
        console.log(`Loaded temperature scaling from model: T = ${TEMPERATURE}`);
      }
      if (modelMetadata.version) {
        console.log(`Model version: ${modelMetadata.version}`);
      }
    }
    
    return model;
  });
  
  return modelPromise;
}



// Backend de cómputo activo de TF.js ('webgl' = GPU, 'cpu' = fallback).
// Útil como dato técnico en la UI.
export function getBackend() {
  return tf.getBackend();
}

/**
 * Clasifica una imagen de lesión cutánea.
 *
 * COHERENCIA CON EL ENTRENAMIENTO: el preprocesado (resize 224×224 + ÷255)
 * debe coincidir EXACTAMENTE con el del notebook. Usamos Rescaling(1./255)
 * y NO vgg16.preprocess_input (que restaría la media de ImageNet por canal).
 * Si se cambia el preprocesado del notebook, este código debe actualizarse
 * o las predicciones serán silenciosamente incorrectas.
 *
 * Devuelve un objeto con el score crudo y el calibrado para que la UI pueda
 * mostrar ambos con la etiqueta correcta (útil para el logit).
 *
 * @param {HTMLImageElement} imgElement imagen ya decodificada (img.complete === true)
 * @returns {Promise<{raw: number, calibrated: number}>}
 */
export async function predictImage(imgElement) {
  const model = await loadModel();

  // fromPixelsAsync (TF.js 4+) reemplaza a fromPixels deprecated.
  // Creamos el tensor manualmente y usamos try/finally para limpieza.
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
    return { raw: safe, calibrated: calibrate(safe) };
  } finally {
    pixels.dispose();
    input?.dispose();
    output?.dispose();
  }
}
