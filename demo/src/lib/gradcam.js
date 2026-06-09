import * as tf from '@tensorflow/tfjs';

// Capa convolucional cuyo gradiente queremos visualizar. block5_conv3 es la
// última conv del backbone VGG16, justo antes del pool global; coincide con
// la capa que el notebook usa para Grad-CAM++ por consistencia.
// Posibles nombres en el modelo convertido (TF.js a veces renombra capas).
const TARGET_LAYER_CANDIDATES = [
  'block5_conv3',
  'block5_conv3_1',  // posible sufijo en conversión
  'conv5_block3_conv',  // formato ResNet-style
];
const INPUT_SIZE = 224;

// El "modelo cabeza" (activations -> prediction) se reconstruye partiendo
// las capas finales del modelo original. Es caro de instanciar y no cambia,
// así que lo cacheamos por modelo (usando WeakMap por si hay reload).
const splitCache = new WeakMap();

function findTargetLayer(model) {
  for (const name of TARGET_LAYER_CANDIDATES) {
    try {
      return model.getLayer(name);
    } catch {
      // continuar con el siguiente candidato
    }
  }
  // Fallback: buscar la última capa convolucional 4D
  const convLayers = model.layers.filter(l => l.output && l.output.shape?.length === 4);
  if (convLayers.length > 0) {
    console.warn(`Grad-CAM: capa objetivo no encontrada, usando última conv: ${convLayers[convLayers.length - 1].name}`);
    return convLayers[convLayers.length - 1];
  }
  throw new Error('No se encontró ninguna capa convolucional 4D para Grad-CAM');
}

function getSplitModels(model) {
  const cached = splitCache.get(model);
  if (cached) return cached;

  const target = findTargetLayer(model);
  // 1) input -> activations del último conv
  const actModel = tf.model({ inputs: model.inputs, outputs: target.output });

  // 2) activations -> prediction. Reaplica todas las capas posteriores a la
  // capa objetivo sobre un nuevo input. Estas capas se reutilizan (mismos
  // weights), pero en tiempo de inferencia es seguro: Dropout con
  // training=false (default) es no-op y el resto son deterministas.
  const layers = model.layers;
  const idx = layers.findIndex((l) => l.name === target.name);
  const tail = layers.slice(idx + 1);

  const actShape = target.output.shape.slice(1); // [H, W, C]
  const actInput = tf.input({ shape: actShape });
  let x = actInput;
  for (const layer of tail) {
    x = layer.apply(x);
  }
  const clsModel = tf.model({ inputs: actInput, outputs: x });

  const out = { actModel, clsModel };
  splitCache.set(model, out);
  return out;
}

/**
 * Calcula Grad-CAM para una imagen ya preprocesada.
 *
 * @param {tf.LayersModel} model         modelo completo
 * @param {HTMLImageElement} imgElement  imagen decodificada
 * @returns {Promise<Float32Array>}      heatmap normalizado [0, 1] de
 *                                       INPUT_SIZE*INPUT_SIZE valores.
 */
export async function computeGradCAM(model, imgElement) {
  const { actModel, clsModel } = getSplitModels(model);

  // Mismo preprocesado que predictImage(): resize 224×224 + /255 + batch.
  // fromPixelsAsync (TF.js 4+) reemplaza a fromPixels deprecated.
  const pixels = await tf.browser.fromPixelsAsync(imgElement);
  const input = tf.tidy(() =>
    pixels
      .resizeBilinear([INPUT_SIZE, INPUT_SIZE])
      .toFloat()
      .div(255)
      .expandDims(0)
  );

  // Calcula activations + gradiente y normaliza. Toda la cadena dentro de
  // tf.tidy salvo el heatmap final, que devolvemos.
  let heatmapData;
  let heatmap;
  let activations;
  let grads;
  try {
    activations = actModel.predict(input);

    // d(prediction) / d(activations). tf.grad espera una función escalar.
    const gradFn = tf.grad((a) => clsModel.predict(a).squeeze());
    grads = gradFn(activations);

    heatmap = tf.tidy(() => {
      // alpha_k = mean over (batch, H, W) de los gradientes -> [C]
      const pooledGrads = grads.mean([0, 1, 2]);
      // Pesar las activations por sus gradientes promedio
      const weighted = activations.squeeze().mul(pooledGrads); // [H, W, C]
      // Sumar canales, aplicar ReLU
      const camRaw = weighted.sum(-1).relu(); // [H, W]
      // Normalizar a [0, 1]. Añadimos eps para evitar 0/0 si el heatmap es plano.
      const minV = camRaw.min();
      const maxV = camRaw.max();
      const camNorm = camRaw.sub(minV).div(maxV.sub(minV).add(1e-8));
      // Resize a tamaño de entrada
      const resized = tf.image.resizeBilinear(
        camNorm.expandDims(0).expandDims(-1),
        [INPUT_SIZE, INPUT_SIZE],
      );
      return resized.squeeze(); // [H, W]
    });

    heatmapData = await heatmap.data();
  } finally {
    activations?.dispose();
    grads?.dispose();
    pixels.dispose();
    input.dispose();
    heatmap?.dispose();
  }

  return heatmapData;
}

/**
 * Pinta un heatmap [0, 1] sobre un canvas usando una colormap "inferno-ish":
 * azul oscuro → magenta → naranja → amarillo. La transparencia escala con
 * la magnitud para que las zonas bajas dejen ver la imagen original.
 */
export function paintHeatmap(canvas, heatmap, width = INPUT_SIZE, height = INPUT_SIZE) {
  if (!heatmap) return;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const data = img.data;
  for (let i = 0; i < heatmap.length; i++) {
    const v = Math.max(0, Math.min(1, heatmap[i]));
    // Colormap aproximada: paso por (azul→magenta→naranja→amarillo). Tres
    // tramos lineales sobre v ∈ [0, 1/3, 2/3, 1].
    let r, g, b;
    if (v < 1 / 3) {
      const t = v * 3;
      r = Math.round(20 + 175 * t); // 20 → 195
      g = 0;
      b = Math.round(50 + 100 * (1 - t)); // 150 → 50
    } else if (v < 2 / 3) {
      const t = (v - 1 / 3) * 3;
      r = Math.round(195 + 60 * t); // 195 → 255
      g = Math.round(80 * t); // 0 → 80
      b = Math.round(50 * (1 - t)); // 50 → 0
    } else {
      const t = (v - 2 / 3) * 3;
      r = 255;
      g = Math.round(80 + 175 * t); // 80 → 255
      b = Math.round(40 * t); // 0 → 40
    }
    const a = Math.round(210 * v); // 0 → ~0.82
    const k = i * 4;
    data[k] = r;
    data[k + 1] = g;
    data[k + 2] = b;
    data[k + 3] = a;
  }
  ctx.putImageData(img, 0, 0);
}
