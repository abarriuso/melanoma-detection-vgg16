// Mock de @tensorflow/tfjs para tests unitarios.
// Simula la API de TF.js con objetos mock que retornan valores predecibles.

function createMockTensor() {
  return {
    resizeBilinear: function () { return this; },
    toFloat: function () { return this; },
    div: function () { return this; },
    expandDims: function () { return this; },
    squeeze: function () { return this; },
    mul: function () { return this; },
    sum: function () { return this; },
    relu: function () { return this; },
    mean: function () { return this; },
    sub: function () { return this; },
    min: function () { return createMockScalar(0); },
    max: function () { return createMockScalar(1); },
    data: async () => new Float32Array([0.8]),
    dispose: vi.fn(),
    shape: [1, 224, 224, 3],
  };
}

function createMockScalar(value) {
  return {
    data: async () => new Float32Array([value]),
    dispose: vi.fn(),
  };
}

const mockModel = {
  predict: vi.fn(() => createMockTensor()),
  userDefinedMetadata: { version: '1.0.0', temperature: 0.902 },
  inputs: [{ shape: [null, 224, 224, 3] }],
  outputs: [{ shape: [null, 1] }],
  layers: [
    { name: 'block5_conv3', output: { shape: [null, 14, 14, 512] } },
    { name: 'global_average_pooling2d', output: { shape: [null, 512] } },
    { name: 'dense', output: { shape: [null, 256] } },
    { name: 'dropout', output: { shape: [null, 256] } },
    { name: 'dense_1', output: { shape: [null, 1] } },
  ],
  getLayer: vi.fn((name) => ({
    name,
    output: { shape: [null, 14, 14, 512] },
  })),
};

const tf = {
  loadLayersModel: vi.fn(async () => mockModel),
  browser: {
    fromPixelsAsync: vi.fn(async () => createMockTensor()),
  },
  tidy: vi.fn((fn) => fn()),
  grad: vi.fn((fn) => {
    return (x) => {
      fn(x);
      return createMockTensor();
    };
  }),
  model: vi.fn(() => mockModel),
  input: vi.fn(() => ({ apply: vi.fn(() => createMockTensor()) })),
  getBackend: vi.fn(() => 'cpu'),
  setBackend: vi.fn(async () => {}),
  ready: vi.fn(async () => {}),
  image: {
    resizeBilinear: vi.fn(() => createMockTensor()),
  },
};

export default tf;
export const loadLayersModel = tf.loadLayersModel;
export const browser = tf.browser;
export const tidy = tf.tidy;
export const grad = tf.grad;
export const model = tf.model;
export const input = tf.input;
export const getBackend = tf.getBackend;
export const setBackend = tf.setBackend;
export const ready = tf.ready;
export const image = tf.image;
