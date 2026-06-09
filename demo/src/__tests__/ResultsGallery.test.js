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
