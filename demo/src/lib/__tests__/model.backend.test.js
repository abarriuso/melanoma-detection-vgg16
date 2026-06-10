import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@tensorflow/tfjs', () => import('../../../__mocks__/@tensorflow/tfjs.js'));

describe('backend selection', () => {
  let tf;
  let loadModel;

  beforeEach(async () => {
    vi.resetModules();
    const mod = await import('../model.js');
    loadModel = mod.loadModel;
    tf = await import('@tensorflow/tfjs');
    tf.setBackend.mockReset();
    tf.ready.mockReset();
  });

  it('intenta WebGL primero cuando está disponible', async () => {
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    expect(tf.setBackend).toHaveBeenCalledWith('webgl');
  });

  it('usa solo WebGL si funciona (no cae a CPU)', async () => {
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    expect(tf.setBackend).toHaveBeenCalledTimes(1);
    expect(tf.setBackend).toHaveBeenCalledWith('webgl');
  });

  it('cae a CPU cuando WebGL lanza error', async () => {
    tf.setBackend.mockImplementation(async (backend) => {
      if (backend === 'webgl') throw new Error('WebGL no disponible');
    });
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    const backends = tf.setBackend.mock.calls.map(c => c[0]);
    expect(backends).toEqual(['webgl', 'cpu']);
  });

  it('cae a CPU cuando WebGL lanza TypeError', async () => {
    tf.setBackend.mockImplementation(async (backend) => {
      if (backend === 'webgl') throw new TypeError('Cannot read properties of null');
    });
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    const backends = tf.setBackend.mock.calls.map(c => c[0]);
    expect(backends).toEqual(['webgl', 'cpu']);
  });

  it('cae a CPU cuando WebGL rechaza la promesa', async () => {
    tf.setBackend.mockImplementation((backend) => {
      if (backend === 'webgl') return Promise.reject(new Error('WebGL fail'));
      return Promise.resolve();
    });
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    const backends = tf.setBackend.mock.calls.map(c => c[0]);
    expect(backends).toEqual(['webgl', 'cpu']);
  });

  it('nunca intenta webgpu', async () => {
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    const backends = tf.setBackend.mock.calls.map(c => c[0]);
    expect(backends).not.toContain('webgpu');
  });

  it('carga el modelo correctamente con CPU (fallback)', async () => {
    tf.setBackend.mockImplementation(async (backend) => {
      if (backend === 'webgl') throw new Error('WebGL fail');
    });
    tf.ready.mockImplementation(async () => {});
    const model = await loadModel('vgg16');
    expect(model).toBeDefined();
    expect(model.predict).toBeDefined();
  });

  it('carga el modelo correctamente con WebGL', async () => {
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(async () => {});
    const model = await loadModel('vgg16');
    expect(model).toBeDefined();
    expect(model.predict).toBeDefined();
  });

  it('solo inicializa backend una vez (backendPromise cache)', async () => {
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(async () => {});
    await loadModel('vgg16');
    // La segunda llamada usa el backendPromise cacheado
    const oldCalls = tf.setBackend.mock.calls.length;
    await loadModel('vgg16');
    expect(tf.setBackend.mock.calls.length).toBe(oldCalls);
  });

  it('ready() se llama tras setBackend cuando WebGL funciona', async () => {
    const readyMock = vi.fn(async () => {});
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(readyMock);
    await loadModel('vgg16');
    // webgl → ready (1 llamada a ready)
    expect(readyMock).toHaveBeenCalledTimes(1);
  });

  it('ready() se llama tras cada setBackend en fallback', async () => {
    const readyMock = vi.fn(async () => {});
    tf.setBackend.mockImplementation(async (backend) => {
      if (backend === 'webgl') throw new Error('WebGL fail');
    });
    tf.ready.mockImplementation(readyMock);
    await loadModel('vgg16');
    // catch → setBackend('cpu') → ready (1 llamada a ready, solo para cpu)
    expect(readyMock).toHaveBeenCalledTimes(1);
  });

  it('setBackend("cpu") nunca lanza', async () => {
    tf.setBackend.mockImplementation(async (backend) => {
      if (backend === 'webgl') throw new Error('WebGL fail');
    });
    tf.ready.mockImplementation(async () => {});
    await expect(loadModel('vgg16')).resolves.not.toThrow();
  });

  it('predictImage funciona con WebGL', async () => {
    tf.setBackend.mockImplementation(async () => {});
    tf.ready.mockImplementation(async () => {});
    const { predictImage } = await import('../model.js');
    const img = { complete: true, naturalWidth: 224, naturalHeight: 224 };
    const result = await predictImage(img, 'vgg16');
    expect(result).toHaveProperty('raw');
    expect(result).toHaveProperty('calibrated');
  });

  it('predictImage funciona con CPU (fallback)', async () => {
    tf.setBackend.mockImplementation(async (backend) => {
      if (backend === 'webgl') throw new Error('WebGL fail');
    });
    tf.ready.mockImplementation(async () => {});
    const { predictImage } = await import('../model.js');
    const img = { complete: true, naturalWidth: 224, naturalHeight: 224 };
    const result = await predictImage(img, 'vgg16');
    expect(result).toHaveProperty('raw');
    expect(result).toHaveProperty('calibrated');
  });
});
