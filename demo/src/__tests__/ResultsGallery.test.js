import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('@tensorflow/tfjs', () => import('../../__mocks__/@tensorflow/tfjs.js'));

describe('shuffle', () => {
  let shuffle;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../ResultsGallery.jsx');
    shuffle = mod.shuffle;
  });

  it('returns array of same length', () => {
    const arr = [1, 2, 3, 4, 5];
    expect(shuffle(arr)).toHaveLength(5);
  });

  it('does not mutate original array', () => {
    const arr = [1, 2, 3];
    const copy = [...arr];
    shuffle(arr);
    expect(arr).toEqual(copy);
  });

  it('contains all original elements', () => {
    const arr = [1, 2, 3, 4, 5];
    const result = shuffle(arr);
    expect(result.sort()).toEqual(arr.sort());
  });

  it('handles empty array', () => {
    expect(shuffle([])).toEqual([]);
  });

  it('handles single element', () => {
    expect(shuffle([42])).toEqual([42]);
  });
});

describe('loadScoresCache', () => {
  let loadScoresCache;

  beforeEach(async () => {
    vi.resetModules();
    // Mock localStorage
    const storage = {};
    const mockLocalStorage = {
      getItem: vi.fn((key) => storage[key] ?? null),
      setItem: vi.fn((key, val) => { storage[key] = val; }),
      removeItem: vi.fn((key) => { delete storage[key]; }),
    };
    vi.stubGlobal('localStorage', mockLocalStorage);
    const mod = await import('../ResultsGallery.jsx');
    loadScoresCache = mod.loadScoresCache;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns empty object when nothing cached', () => {
    expect(loadScoresCache()).toEqual({});
  });

  it('returns parsed object when valid JSON exists', () => {
    localStorage.getItem.mockReturnValue('{"img1.jpg": 0.8}');
    expect(loadScoresCache()).toEqual({ 'img1.jpg': 0.8 });
  });

  it('returns empty object on corrupt JSON', () => {
    localStorage.getItem.mockReturnValue('{invalid json');
    expect(loadScoresCache()).toEqual({});
  });

  it('descarta un array (no es un mapa de scores)', () => {
    localStorage.getItem.mockReturnValue('[0.1, 0.2]');
    expect(loadScoresCache()).toEqual({});
  });

  it('descarta scores fuera de [0,1]', () => {
    localStorage.getItem.mockReturnValue('{"a.jpg": -0.5, "b.jpg": 1.5, "ok.jpg": 0.5}');
    expect(loadScoresCache()).toEqual({ 'ok.jpg': 0.5 });
  });

  it('descarta scores que no son números', () => {
    localStorage.getItem.mockReturnValue('{"a.jpg": "0.9", "b.jpg": null, "c.jpg": {}, "ok.jpg": 0.7}');
    expect(loadScoresCache()).toEqual({ 'ok.jpg': 0.7 });
  });

  it('descarta NaN e Infinity', () => {
    // NaN/Infinity no son JSON válidos, llegan como null o como strings
    localStorage.getItem.mockReturnValue('{"a.jpg": null, "ok.jpg": 1}');
    expect(loadScoresCache()).toEqual({ 'ok.jpg': 1 });
  });

  it('no hereda propiedades de Object.prototype', () => {
    localStorage.getItem.mockReturnValue('{"ok.jpg": 0.5}');
    const cache = loadScoresCache();
    // Con un objeto normal, cache['toString'] devolvería la función heredada
    // y el chequeo `!= null` de allScored daría un falso positivo.
    expect(cache['toString']).toBeUndefined();
  });
});

describe('saveScoresCache', () => {
  let saveScoresCache;

  beforeEach(async () => {
    vi.resetModules();
    const storage = {};
    const mockLocalStorage = {
      getItem: vi.fn((key) => storage[key] ?? null),
      setItem: vi.fn((key, val) => { storage[key] = val; }),
    };
    vi.stubGlobal('localStorage', mockLocalStorage);
    const mod = await import('../ResultsGallery.jsx');
    saveScoresCache = mod.saveScoresCache;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns true on success', () => {
    expect(saveScoresCache({ 'img1.jpg': 0.9 })).toBe(true);
  });

  it('returns false when localStorage is full', () => {
    localStorage.setItem.mockImplementation(() => { throw new Error('quota'); });
    expect(saveScoresCache({ 'img1.jpg': 0.9 })).toBe(false);
  });
});
