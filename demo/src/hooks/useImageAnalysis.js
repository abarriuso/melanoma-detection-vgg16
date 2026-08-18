import { useState, useCallback, useRef, useEffect } from 'react';
import { UMBRAL } from '../lib/constants';

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export function useImageAnalysis(predictFn) {
  const [imageURL, setImageURL] = useState(null);
  const [imageError, setImageError] = useState(false);
  const [fileError, setFileError] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const runTokenRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const clearImage = useCallback(() => {
    runTokenRef.current++;
    if (imageURL?.startsWith('blob:')) URL.revokeObjectURL(imageURL);
    setImageURL(null);
    setImageError(false);
    setFileError(null);
    setResult(null);
    setPredicting(false);
    setPredictionError(null);
  }, [imageURL]);

  const setImage = useCallback((url, { auto = false } = {}) => {
    setResult(null);
    setImageError(false);
    setFileError(null);
    setPredictionError(null);
    setImageURL((prev) => {
      if (prev && prev.startsWith('blob:') && prev !== url) {
        URL.revokeObjectURL(prev);
      }
      return url;
    });
    setAutoRun(auto);
  }, []);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setFileError('Formato no soportado. Usa JPEG, PNG o WebP.');
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      const mb = (file.size / (1024 * 1024)).toFixed(1);
      setFileError(`Imagen demasiado grande (${mb} MB). Máximo 10 MB.`);
      return;
    }
    const img = new Image();
    const objectURL = URL.createObjectURL(file);
    img.onload = () => {
      const { naturalWidth: w, naturalHeight: h } = img;
      if (w < 16 || h < 16) {
        URL.revokeObjectURL(objectURL);
        setFileError(`Imagen demasiado pequeña (${w}×${h}). Mínimo 16×16 píxeles.`);
        return;
      }
      if (w > 4096 || h > 4096) {
        URL.revokeObjectURL(objectURL);
        setFileError(`Imagen demasiado grande (${w}×${h}). Máximo 4096×4096 píxeles.`);
        return;
      }
      setFileError(null);
      setImage(objectURL, { auto: true });
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectURL);
      setFileError('No se pudo decodificar la imagen.');
    };
    img.src = objectURL;
  }, [setImage]);

  const analyze = useCallback(async (imgElement, modelId, modelStatus) => {
    if (!imgElement || modelStatus !== 'ready') return;
    const myToken = ++runTokenRef.current;
    setPredicting(true);
    setPredictionError(null);
    try {
      const t0 = performance.now();
      const { raw, calibrated } = await predictFn(imgElement, modelId);
      const ms = Math.round(performance.now() - t0);
      if (myToken !== runTokenRef.current || !mountedRef.current) return;
      const esMaligno = calibrated >= UMBRAL;
      const logit = Math.log(raw / (1 - raw));
      setResult({
        score: calibrated,
        logit,
        ms,
        label: esMaligno ? 'Maligno' : 'Benigno',
        confidence: esMaligno ? calibrated : 1 - calibrated,
        esMaligno,
      });
    } catch (err) {
      console.error('Error en la predicción:', err);
      if (mountedRef.current) {
        setPredictionError(
          err?.message === 'timeout'
            ? 'El análisis está tardando demasiado (posible problema con la GPU del navegador). Inténtalo de nuevo.'
            : 'Error al analizar la imagen. Inténtalo de nuevo.',
        );
      }
    } finally {
      if (mountedRef.current && myToken === runTokenRef.current) setPredicting(false);
    }
  }, [predictFn]);

  return {
    imageURL,
    imageError,
    fileError,
    predicting,
    predictionError,
    result,
    dragActive,
    setDragActive,
    autoRun,
    setAutoRun,
    setImage,
    setImageError,
    clearImage,
    handleFile,
    analyze,
    runTokenRef,
    mountedRef,
  };
}
