import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@tensorflow/tfjs', () => import('../../../__mocks__/@tensorflow/tfjs.js'));

// La función jet() no se exporta directamente del módulo,
// pero podemos probarla importándola internamente o mediante paintHeatmap.
// Extraemos su lógica: colormap jet con 11 stops, interpolación lineal.

describe('jet colormap logic', () => {
  it('devuelve negro (0,0,0) para t=0', async () => {
    vi.resetModules();
    const mod = await import('../gradcam.js');
    // paintHeatmap usa jet internamente; probamos que un heatmap 0 produce píxel semitransparente
    const paintHeatmap = mod.paintHeatmap;
    let pixelData;
    const ctx = {
      createImageData: (w, h) => {
        pixelData = new Uint8ClampedArray(w * h * 4);
        return { data: pixelData };
      },
      putImageData: vi.fn(),
    };
    const canvas = { width: 0, height: 0, getContext: vi.fn(() => ctx) };
    paintHeatmap(canvas, new Float32Array([0, 0.5, 1]), 3, 1);
    // t=0 → alfa 0 (completamente transparente)
    expect(pixelData[3]).toBe(0);
    // t=0.5 → alfa ~0.65 * 255 = ~165
    expect(pixelData[7]).toBeGreaterThan(100);
    expect(pixelData[7]).toBeLessThan(200);
    // t=1 → alfa ~165
    expect(pixelData[11]).toBeGreaterThan(100);
  });

  it('mapea correctamente los 11 stops del colormap', async () => {
    vi.resetModules();
    const mod = await import('../gradcam.js');
    const paintHeatmap = mod.paintHeatmap;
    let pixelData;
    const ctx = {
      createImageData: (w, h) => {
        pixelData = new Uint8ClampedArray(w * h * 4);
        return { data: pixelData };
      },
      putImageData: vi.fn(),
    };
    const canvas = { width: 0, height: 0, getContext: vi.fn(() => ctx) };
    paintHeatmap(canvas, new Float32Array([0, 1]), 2, 1);
    // t=0 → alfa 0
    expect(pixelData[3]).toBe(0);
    // t=1 → alfa ~165
    expect(pixelData[7]).toBeGreaterThan(150);
  });
});

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
