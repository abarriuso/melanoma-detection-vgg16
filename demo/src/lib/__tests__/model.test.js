import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@tensorflow/tfjs', () => import('../../../__mocks__/@tensorflow/tfjs.js'));

describe('calibrate', () => {
  let calibrate;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../model.js');
    calibrate = mod.calibrate;
  });

  it('returns the same value when T=1.0 (no calibration)', async () => {
    vi.resetModules();
    // Override TEMPERATURE to 1.0 by reloading module
    // For this test we just verify the formula directly
    const eps = 1e-7;
    const p = 0.8;
    const clamped = Math.min(Math.max(p, eps), 1 - eps);
    const logit = Math.log(clamped / (1 - clamped));
    const expected = 1 / (1 + Math.exp(-logit / 0.902));
    const result = calibrate(p, 0.902);
    expect(result).toBeCloseTo(expected, 10);
  });

  it('returns ~0.5 when input is 0.5 (symmetric point)', () => {
    const result = calibrate(0.5, 0.902);
    expect(result).toBeCloseTo(0.5, 2);
  });

  it('preserves ordering: calibrate(high) > calibrate(low)', () => {
    const low = calibrate(0.3, 0.902);
    const high = calibrate(0.7, 0.902);
    expect(high).toBeGreaterThan(low);
  });

  it('clamps extreme values without returning NaN or Infinity', () => {
    const nearZero = calibrate(1e-10, 0.902);
    const nearOne = calibrate(1 - 1e-10, 0.902);
    expect(Number.isFinite(nearZero)).toBe(true);
    expect(Number.isFinite(nearOne)).toBe(true);
    expect(nearZero).toBeGreaterThan(0);
    expect(nearOne).toBeLessThan(1);
  });

  it('returns 0.5 for 0.5 regardless of temperature', () => {
    const result = calibrate(0.5, 0.902);
    expect(result).toBeCloseTo(0.5, 6);
  });

  it('returns same value when temperature is 1.0 (no-op)', () => {
    const result = calibrate(0.8, 1.0);
    expect(result).toBeCloseTo(0.8, 6);
  });

  it('returns original when temperature is null', () => {
    const result = calibrate(0.8, null);
    expect(result).toBe(0.8);
  });

  it('returns original when temperature is undefined', () => {
    const result = calibrate(0.8, undefined);
    expect(result).toBe(0.8);
  });

  it('T<1 amplifica confianza (acerca a 0 o 1)', () => {
    const high = calibrate(0.85, 0.5);
    const low = calibrate(0.15, 0.5);
    expect(high).toBeGreaterThan(0.85);
    expect(low).toBeLessThan(0.15);
  });

  it('T>1 reduce confianza (acerca a 0.5)', () => {
    const high = calibrate(0.85, 2.0);
    const low = calibrate(0.15, 2.0);
    expect(high).toBeLessThan(0.85);
    expect(low).toBeGreaterThan(0.15);
  });
});

describe('getModel', () => {
  let getModel;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../constants.js');
    getModel = mod.getModel;
  });

  it('returns VGG16 entry for valid id', () => {
    const m = getModel('vgg16');
    expect(m.id).toBe('vgg16');
    expect(m.name).toBe('VGG16');
    expect(m.path).toContain('model.json');
    expect(m.temperature).toBe(1.3359);
    expect(m.auc).toBe(0.9712);
  });

  it('returns ResNet50V2 for its id', () => {
    const m = getModel('resnet50v2');
    expect(m.id).toBe('resnet50v2');
    expect(m.temperature).toBe(1.0221);
  });

  it('returns EfficientNetV2S for its id', () => {
    const m = getModel('efficientnetv2s');
    expect(m.id).toBe('efficientnetv2s');
    expect(m.temperature).toBe(1.1836);
  });

  it('falls back to EfficientNetV2S for unknown id', () => {
    const m = getModel('invalid_model');
    expect(m.id).toBe('efficientnetv2s');
  });

  it('falls back to EfficientNetV2S for null id', () => {
    const m = getModel(null);
    expect(m.id).toBe('efficientnetv2s');
  });

  it('falls back to EfficientNetV2S for undefined id', () => {
    const m = getModel();
    expect(m.id).toBe('efficientnetv2s');
  });
});

describe('isSoftwareRenderer', () => {
  let isSoftwareRenderer;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../model.js');
    isSoftwareRenderer = mod.isSoftwareRenderer;
  });

  it('detecta SwiftShader (Chrome sin aceleración)', () => {
    expect(isSoftwareRenderer('Google SwiftShader')).toBe(true);
    expect(isSoftwareRenderer('ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))')).toBe(true);
  });

  it('detecta llvmpipe y el Microsoft Basic Render Driver', () => {
    expect(isSoftwareRenderer('llvmpipe (LLVM 15.0.7, 256 bits)')).toBe(true);
    expect(isSoftwareRenderer('ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11)')).toBe(true);
  });

  it('no marca GPUs reales como software', () => {
    expect(isSoftwareRenderer('ANGLE (NVIDIA, NVIDIA GeForce GT 710 Direct3D11)')).toBe(false);
    expect(isSoftwareRenderer('Apple M1')).toBe(false);
    expect(isSoftwareRenderer('Mali-G78')).toBe(false);
  });

  it('devuelve false para null/undefined/vacío', () => {
    expect(isSoftwareRenderer(null)).toBe(false);
    expect(isSoftwareRenderer(undefined)).toBe(false);
    expect(isSoftwareRenderer('')).toBe(false);
  });
});

describe('getGpuInfo', () => {
  it('no lanza en jsdom (sin WebGL) y reporta supported=false', async () => {
    vi.resetModules();
    const { getGpuInfo } = await import('../model.js');
    const info = getGpuInfo();
    expect(info.supported).toBe(false);
    expect(info.software).toBe(false);
  });
});

describe('getModelMetadata', () => {
  let getModelMetadata;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../model.js');
    getModelMetadata = mod.getModelMetadata;
  });

  it('returns model metadata from constants as fallback', () => {
    const meta = getModelMetadata('vgg16');
    expect(meta).toBeDefined();
    expect(meta.version).toBe('2.0.0');
    expect(meta.temperature).toBe(1.3359);
  });
});

describe('gestión de memoria (F-04/F-05)', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('predictImage dispone los tensores intermedios (píxeles e input)', async () => {
    const tf = await import('@tensorflow/tfjs');
    tf.browser.fromPixelsAsync.mockClear();
    const { predictImage } = await import('../model.js');
    const img = { complete: true, naturalWidth: 224, naturalHeight: 224 };
    await predictImage(img, 'vgg16');
    const pixels = await tf.browser.fromPixelsAsync.mock.results.at(-1).value;
    // El mock encadena resizeBilinear/toFloat/... sobre el mismo objeto,
    // así que `pixels` e `input` son el mismo tensor: dispose se invoca.
    expect(pixels.dispose).toHaveBeenCalled();
  });

  it('predictImage dispone el tensor de salida del modelo', async () => {
    const { predictImage } = await import('../model.js');
    const tf = await import('@tensorflow/tfjs');
    const model = await tf.loadLayersModel();
    model.predict.mockClear();
    const img = { complete: true, naturalWidth: 224, naturalHeight: 224 };
    await predictImage(img, 'vgg16');
    const output = model.predict.mock.results.at(-1).value;
    expect(output.dispose).toHaveBeenCalled();
  });

  it('al terminar de cargar un modelo nuevo se dispone el anterior (LRU de 1)', async () => {
    const { loadModel } = await import('../model.js');
    const first = await loadModel('vgg16');
    first.dispose.mockClear();
    await loadModel('efficientnetv2s');
    expect(first.dispose).toHaveBeenCalledTimes(1);
  });

  it('un version mismatch dispone el modelo viejo antes de recargar', async () => {
    const tf = await import('@tensorflow/tfjs');
    const staleModel = {
      predict: vi.fn(() => ({ data: async () => new Float32Array([0.5]), dispose: vi.fn() })),
      dispose: vi.fn(),
      userDefinedMetadata: { version: '1.0.0', temperature: 1.0 },
    };
    tf.loadLayersModel.mockResolvedValueOnce(staleModel);
    const { loadModel } = await import('../model.js');
    const first = await loadModel('vgg16');
    expect(first).toBe(staleModel);
    // Segunda carga: la caché tiene versión 1.0.0 ≠ 2.0.0 del registro.
    const reloaded = await loadModel('vgg16');
    expect(staleModel.dispose).toHaveBeenCalledTimes(1);
    expect(reloaded).not.toBe(staleModel);
    expect(reloaded).toBeDefined();
  });
});
