import { useState, useEffect, useRef, useCallback } from 'react';
import { loadModel, getBackend, getGpuInfo, setActiveModelId } from '../lib/model';

const PREDICT_TIMEOUT_MS = 60_000;

function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}

export function useModel(modelId) {
  const [modelStatus, setModelStatus] = useState('loading');
  const [progress, setProgress] = useState(0);
  const [backend, setBackend] = useState('—');
  const [slowGpu, setSlowGpu] = useState(false);
  const mountedRef = useRef(true);

  useEffect(() => () => { mountedRef.current = false; }, []);

  useEffect(() => {
    let mounted = true;
    setModelStatus('loading');
    setProgress(0);
    setActiveModelId(modelId);
    loadModel(modelId, (p) => mounted && setProgress(Math.round(p * 100)))
      .then(() => {
        if (!mounted) return;
        const activeBackend = getBackend();
        setBackend(activeBackend);
        const gpu = getGpuInfo();
        setSlowGpu(activeBackend === 'cpu' || !gpu.supported || gpu.software);
        setModelStatus('ready');
      })
      .catch((err) => {
        console.error('Error cargando el modelo:', err);
        if (mounted) setModelStatus('error');
      });
    return () => { mounted = false; };
  }, [modelId]);

  const predict = useCallback(async (imgElement, id) => {
    const { predictImage } = await import('../lib/model');
    return withTimeout(predictImage(imgElement, id), PREDICT_TIMEOUT_MS, 'timeout');
  }, []);

  return {
    modelStatus,
    progress,
    backend,
    slowGpu,
    predict,
  };
}
