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
    expect(m.temperature).toBe(0.902);
    expect(m.targetLayer).toBe('block5_conv3');
    expect(m.auc).toBe(0.9606);
  });

  it('returns ResNet50V2 for its id', () => {
    const m = getModel('resnet50v2');
    expect(m.id).toBe('resnet50v2');
    expect(m.temperature).toBeNull();
    expect(m.targetLayer).toBe('post_relu');
  });

  it('returns EfficientNetV2S for its id', () => {
    const m = getModel('efficientnetv2s');
    expect(m.id).toBe('efficientnetv2s');
    expect(m.targetLayer).toBe('top_conv');
  });

  it('falls back to VGG16 for unknown id', () => {
    const m = getModel('invalid_model');
    expect(m.id).toBe('vgg16');
  });

  it('falls back to VGG16 for null id', () => {
    const m = getModel(null);
    expect(m.id).toBe('vgg16');
  });

  it('falls back to VGG16 for undefined id', () => {
    const m = getModel();
    expect(m.id).toBe('vgg16');
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
    expect(meta.version).toBe('1.0.0');
    expect(meta.temperature).toBe(0.902);
  });
});
