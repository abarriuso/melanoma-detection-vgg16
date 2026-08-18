import { describe, it, expect, vi, beforeAll } from 'vitest';

// A diferencia del resto de tests, este necesita el MOTOR REAL de tfjs para
// ejercitar el gradiente (el mock devuelve tensores constantes y no detectaría
// este bug). Solo sustituimos fromPixelsAsync, que requiere un canvas real del
// navegador, por un tensor sintético.
vi.mock('@tensorflow/tfjs', async (importOriginal) => {
  const real = await importOriginal();
  return {
    ...real,
    browser: {
      ...real.browser,
      fromPixelsAsync: async () => real.randomUniform([32, 32, 3], 0, 1, 'float32', 42),
    },
  };
});

import * as tf from '@tensorflow/tfjs';
import { computeGradCAM } from '../gradcam.js';

// Clasificador diminuto con la misma forma que la cabeza de la demo
// (conv → GAP → Dense sigmoide) pero con pesos que SATURAN el sigmoide
// (prob ≈ 1.0). Ese es justo el caso donde derivar la probabilidad en vez del
// logit dejaba el heatmap entero a cero (overlay en blanco).
function buildSaturatedModel() {
  const input = tf.input({ shape: [224, 224, 3] });
  const conv = tf.layers
    .conv2d({
      filters: 4,
      kernelSize: 3,
      padding: 'same',
      activation: 'relu',
      name: 'block5_conv3',
      kernelInitializer: 'ones',
      biasInitializer: 'ones',
    })
    .apply(input);
  const gap = tf.layers.globalAveragePooling2d({}).apply(conv);
  const out = tf.layers
    .dense({
      units: 1,
      activation: 'sigmoid',
      kernelInitializer: tf.initializers.constant({ value: 50 }),
      biasInitializer: tf.initializers.constant({ value: 50 }),
    })
    .apply(gap);
  return tf.model({ inputs: input, outputs: out });
}

describe('computeGradCAM (motor real)', () => {
  beforeAll(async () => {
    await tf.setBackend('cpu');
    await tf.ready();
  });

  it('produce un heatmap NO nulo aunque el sigmoide esté saturado', async () => {
    const model = buildSaturatedModel();

    // El modelo está efectivamente saturado (prob ≈ 1).
    const prob = (await model.predict(tf.ones([1, 224, 224, 3])).data())[0];
    expect(prob).toBeGreaterThan(0.999);

    const heatmap = await computeGradCAM(model, /* imgElement */ {}, undefined);

    expect(heatmap).toHaveLength(224 * 224);

    let max = -Infinity;
    let allFinite = true;
    for (const v of heatmap) {
      if (v > max) max = v;
      if (!Number.isFinite(v)) allFinite = false;
    }
    expect(allFinite).toBe(true);
    // Regresión: derivar la probabilidad saturada dejaba TODO a cero. Al
    // derivar el logit hay señal.
    expect(max).toBeGreaterThan(0);
  }, 30000);
});
