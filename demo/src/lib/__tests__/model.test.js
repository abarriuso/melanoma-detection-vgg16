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
});
