import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@tensorflow/tfjs', () => import('../../../__mocks__/@tensorflow/tfjs.js'));

describe('paintHeatmap', () => {
  let paintHeatmap;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../gradcam.js');
    paintHeatmap = mod.paintHeatmap;
  });

  it('does nothing when heatmap is null', () => {
    const canvas = { width: 0, height: 0, getContext: vi.fn() };
    paintHeatmap(canvas, null);
    expect(canvas.getContext).not.toHaveBeenCalled();
  });

  it('does nothing when heatmap is undefined', () => {
    const canvas = { width: 0, height: 0, getContext: vi.fn() };
    paintHeatmap(canvas, undefined);
    expect(canvas.getContext).not.toHaveBeenCalled();
  });

  it('sets canvas dimensions from heatmap size', () => {
    const putImageData = vi.fn();
    const ctx = {
      createImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
      putImageData,
    };
    const canvas = { width: 0, height: 0, getContext: vi.fn(() => ctx) };
    const heatmap = new Float32Array([0.5]);
    paintHeatmap(canvas, heatmap, 1, 1);
    expect(canvas.width).toBe(1);
    expect(canvas.height).toBe(1);
  });

  it('calls putImageData once', () => {
    const putImageData = vi.fn();
    const ctx = {
      createImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
      putImageData,
    };
    const canvas = { width: 0, height: 0, getContext: vi.fn(() => ctx) };
    const heatmap = new Float32Array([0.5]);
    paintHeatmap(canvas, heatmap, 1, 1);
    expect(putImageData).toHaveBeenCalledTimes(1);
  });

  it('clamps values outside [0,1]', () => {
    let pixelData;
    const ctx = {
      createImageData: (w, h) => {
        pixelData = new Uint8ClampedArray(w * h * 4);
        return { data: pixelData };
      },
      putImageData: vi.fn(),
    };
    const canvas = { width: 0, height: 0, getContext: vi.fn(() => ctx) };
    const heatmap = new Float32Array([-0.5, 1.5]);
    paintHeatmap(canvas, heatmap, 2, 1);
    expect(pixelData[0]).toBeGreaterThanOrEqual(0);
    expect(pixelData[0]).toBeLessThanOrEqual(255);
  });
});
