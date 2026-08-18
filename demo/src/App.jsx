import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { getModel, GITHUB_USER, REPO_NAME, UMBRAL } from './lib/constants';
import { useLocalStorage } from './hooks/useLocalStorage';
import { useModel } from './hooks/useModel';
import { useImageAnalysis } from './hooks/useImageAnalysis';
import ParticleBackground from './components/ParticleBackground';
import Hero from './components/Hero';
import ModelStatusBar from './components/ModelStatusBar';
import GpuWarning from './components/GpuWarning';
import Dropzone from './components/Dropzone';
import ResultCard from './components/ResultCard';
import GradCamToggle from './components/GradCamToggle';
import ModelSelector from './components/ModelSelector';
import ExampleGallery from './components/ExampleGallery';
import Footer from './components/Footer';
import ErrorBoundary from './ErrorBoundary';
import './App.css';

const BASE = import.meta.env.BASE_URL;
const loadGradCAM = () => import('./lib/gradcam');

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function App() {
  // --- Persistencia ---
  const [modelId, setModelId] = useLocalStorage('modelId', 'efficientnetv2s');
  const storedModel = getModel(modelId);
  const validModelId = storedModel.auc != null ? modelId : 'efficientnetv2s';

  // Corregir modelId inválido en localStorage (sin setState durante render)
  useEffect(() => {
    if (validModelId !== modelId) setModelId(validModelId);
  }, [validModelId, modelId, setModelId]);

  // --- Modelo ---
  const { modelStatus, progress, backend, slowGpu, predict } = useModel(validModelId);

  // --- Imagen / Análisis ---
  const {
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
    mountedRef,
  } = useImageAnalysis(predict);

  // --- Grad-CAM ---
  const [showCam, setShowCam] = useState(false);
  const [camBusy, setCamBusy] = useState(false);
  const camCanvasRef = useRef(null);
  const [camRect, setCamRect] = useState(null);

  // --- Refs ---
  const imgRef = useRef(null);
  const inputRef = useRef(null);
  const analizarRef = useRef(null);
  const pendingAutoRef = useRef(false);
  const manifestRef = useRef(null);

  // --- Motion preferences ---
  const prefersReducedMotion = useReducedMotion();
  const fadeUp = prefersReducedMotion
    ? {}
    : { initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, exit: { opacity: 0, y: -6 }, transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] } };
  const fadeIn = prefersReducedMotion
    ? {}
    : { initial: { opacity: 0 }, animate: { opacity: 1 }, exit: { opacity: 0 }, transition: { duration: 0.25 } };

  // --- Ejemplos ---
  const [examples, setExamples] = useState([]);
  const [hasExamplesDataset2, setHasExamplesDataset2] = useState(false);

  const pickExamples = useCallback((d, { random = false } = {}) => {
    const pickN = (arr, real, n = 3) => {
      const pool = random ? shuffle(arr) : arr;
      return pool.slice(0, n).map((name) => ({ real, path: `${BASE}samples/${real}/${name}` }));
    };
    return [...pickN(d.malignant, 'malignant'), ...pickN(d.benign, 'benign')];
  }, []);

  const rotateExamples = useCallback(() => {
    if (manifestRef.current) setExamples(pickExamples(manifestRef.current, { random: true }));
  }, [pickExamples]);

  // --- Carga manifest + resolución ?sample= ---
  useEffect(() => {
    let mounted = true;
    fetch(`${BASE}samples/manifest.json`)
      .then((r) => { if (!r.ok) throw new Error(`manifest ${r.status}`); return r.json(); })
      .then((d) => {
        if (!mounted) return;
        manifestRef.current = d;
        setExamples(pickExamples(d));
        setHasExamplesDataset2(
          (d.malignant || []).some((f) => f.startsWith('isic17_')) ||
          (d.benign || []).some((f) => f.startsWith('isic17_')),
        );

        const params = new URLSearchParams(window.location.search);
        const wanted = params.get('sample');
        if (!wanted) return;
        const valid = /^(melanoma|isic17)_[\w-]+\.(jpe?g|png|webp)$/i.test(wanted);
        if (!valid) return;
        if (!d.malignant || !d.benign) return;
        const real = d.malignant.includes(wanted) ? 'malignant' : d.benign.includes(wanted) ? 'benign' : null;
        if (!real) return;
        setImage(`${BASE}samples/${real}/${wanted}`, { auto: true });
      })
      .catch(() => mounted && setExamples([]));
    return () => { mounted = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [BASE]);

  // --- Análisis ---
  const imagenLista = useCallback(() => !!imgRef.current && imgRef.current.complete && imgRef.current.naturalWidth > 0, []);

  const analizar = useCallback(async () => {
    if (!imagenLista() || modelStatus !== 'ready') return;
    await analyze(imgRef.current, validModelId, modelStatus);
  }, [imagenLista, modelStatus, analyze, validModelId]);
  analizarRef.current = analizar;

  // Retry auto-analysis cuando el modelo termina de cargar
  useEffect(() => {
    if (modelStatus === 'ready' && pendingAutoRef.current && imageURL) {
      pendingAutoRef.current = false;
      analizar();
    }
  }, [modelStatus, imageURL, analizar]);

  // --- Grad-CAM ---
  const renderGradCAM = useCallback(async () => {
    if (!imgRef.current || !camCanvasRef.current) return;
    setCamBusy(true);
    try {
      const { loadModel: loadModelFn } = await import('./lib/model');
      const [{ computeGradCAM, paintHeatmap }, model] = await Promise.all([
        loadGradCAM(),
        loadModelFn(validModelId),
      ]);
      const heatmap = await computeGradCAM(model, imgRef.current, validModelId);
      if (!mountedRef.current) return;
      paintHeatmap(camCanvasRef.current, heatmap, 224, 224);
    } catch (err) {
      console.error('Grad-CAM:', err);
      if (mountedRef.current) {
        setShowCam(false);
      }
    } finally {
      if (mountedRef.current) setCamBusy(false);
    }
  }, [validModelId, mountedRef]);

  useEffect(() => {
    if (!showCam || !result) return;
    renderGradCAM();
  }, [showCam, result, renderGradCAM]);

  // --- Cálculo rect de Grad-CAM ---
  const updateCamRect = useCallback(() => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth || !img.naturalHeight) return;
    const cw = img.clientWidth;
    const ch = img.clientHeight;
    if (!cw || !ch) return;
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    setCamRect({ left: (cw - w) / 2, top: (ch - h) / 2, width: w, height: h });
  }, []);

  useEffect(() => {
    const img = imgRef.current;
    if (!img || !imageURL) return;
    const ro = new ResizeObserver(updateCamRect);
    ro.observe(img);
    return () => ro.disconnect();
  }, [imageURL, updateCamRect]);

  const onImgLoad = () => {
    updateCamRect();
    if (autoRun) {
      setAutoRun(false);
      if (modelStatus === 'ready') {
        analizar();
      } else {
        pendingAutoRef.current = true;
      }
    }
  };

  // --- Drag & Drop ---
  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0]);
  };
  const onDragOver = (e) => { e.preventDefault(); if (!dragActive) setDragActive(true); };

  // --- Atajo de teclado global ---
  const imageURLRef = useRef(null);
  const imageErrorRef = useRef(false);
  const predictingRef = useRef(false);
  imageURLRef.current = imageURL;
  imageErrorRef.current = imageError;
  predictingRef.current = predicting;

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Enter') return;
      const t = e.target;
      if (t?.matches?.('input, textarea, [contenteditable="true"], button')) return;
      if (!imageURLRef.current || imageErrorRef.current || predictingRef.current) return;
      e.preventDefault();
      analizarRef.current?.();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  // --- Cambio de modelo ---
  const handleModelChange = useCallback((id) => {
    if (id !== validModelId) {
      setModelId(id);
      clearImage();
      setShowCam(false);
    }
  }, [validModelId, setModelId, clearImage]);

  // --- Selección de ejemplo ---
  const handleExampleSelect = useCallback((path) => {
    if (path === 'rotate') {
      rotateExamples();
    } else {
      setImage(path, { auto: true });
    }
  }, [rotateExamples, setImage]);

  return (
    <>
      <ParticleBackground />
      <div className="mesh-bg" aria-hidden="true" />
      <div className="grid-pattern" aria-hidden="true" />
      <div className="app" aria-busy={modelStatus === 'loading'}>
        <a href="#main-content" className="skip-link">Saltar al contenido principal</a>

        <nav className="toplinks" aria-label="Enlaces al código">
          <a href={`https://github.com/${GITHUB_USER}/${REPO_NAME}/blob/main/notebooks/entrenamiento_conjunto_kaggle.ipynb`} target="_blank" rel="noreferrer">
            Ver notebook →
          </a>
          <a href={`https://github.com/${GITHUB_USER}/${REPO_NAME}`} target="_blank" rel="noreferrer">
            Código →
          </a>
        </nav>

        <Hero modelName={getModel(validModelId).name} auc={getModel(validModelId).auc} />

        <main id="main-content">
          <ErrorBoundary>
            <div className="main-grid">
              {/* Columna izquierda: Dropzone + resultados */}
              <div className="main-col">
                <section className="panel" aria-labelledby="panel1-title">
                  <div className="panel-head">
                    <span className="panel-idx" aria-hidden="true">01</span>
                    <h2 id="panel1-title">Sube una imagen</h2>
                  </div>
                  <div className="panel-body">
                    <ModelStatusBar status={modelStatus} progress={progress} backend={backend} />

                    <AnimatePresence>
                      {slowGpu && modelStatus === 'ready' && <GpuWarning />}
                    </AnimatePresence>

                    <p className="disclaimer disclaimer--top">
                      Proyecto académico de investigación. No constituye un dispositivo médico
                      ni sustituye la valoración de un profesional sanitario. El modelo tiene
                      una tasa de falsos negativos del ~12%; consulta siempre a un dermatólogo.
                      Diseñado exclusivamente para imágenes dermatoscópicas.
                    </p>

                    <Dropzone
                      imageURL={imageURL}
                      imageError={imageError}
                      dragActive={dragActive}
                      predicting={predicting}
                      showCam={showCam}
                      camRect={camRect}
                      onFile={handleFile}
                      onDrop={onDrop}
                      onDragOver={onDragOver}
                      onDragLeave={() => setDragActive(false)}
                      onClear={clearImage}
                      onImageLoad={onImgLoad}
                      onImageError={() => setImageError(true)}
                      imgRef={imgRef}
                      camCanvasRef={camCanvasRef}
                      inputRef={inputRef}
                      disabled={modelStatus !== 'ready'}
                    />

                    <AnimatePresence>
                      {fileError && (
                        <motion.div className="file-error" role="alert" {...fadeUp}>{fileError}</motion.div>
                      )}
                    </AnimatePresence>

                    <AnimatePresence>
                      {predictionError && (
                        <motion.div className="file-error" role="alert" {...fadeUp}>{predictionError}</motion.div>
                      )}
                    </AnimatePresence>

                    {predicting && (
                      <p className="analyzing-status" role="status" aria-live="polite">
                        Analizando…
                      </p>
                    )}
                    {!predicting && imageURL && !imageError && modelStatus === 'ready' && (result || predictionError) && (
                      <button type="button" className="reanalyze-btn" onClick={analizar}>
                        {predictionError ? 'Reintentar análisis' : 'Analizar de nuevo'}
                      </button>
                    )}

                    <AnimatePresence>
                      {result && (
                        <GradCamToggle
                          showCam={showCam}
                          onToggle={() => setShowCam((v) => !v)}
                          busy={camBusy}
                          disabled={!result}
                        />
                      )}
                    </AnimatePresence>

                    <AnimatePresence>
                      {result && <ResultCard result={result} />}
                    </AnimatePresence>

                    <ExampleGallery
                      examples={examples}
                      onSelect={handleExampleSelect}
                      disabled={modelStatus !== 'ready'}
                    />
                  </div>
                </section>
              </div>

              {/* Columna derecha: Selector de modelo (sticky en desktop) */}
              <div className="main-col">
                <section className="panel panel--sticky" aria-labelledby="panel2-title">
                  <div className="panel-head">
                    <span className="panel-idx" aria-hidden="true">02</span>
                    <h2 id="panel2-title">Configuración</h2>
                  </div>
                  <div className="panel-body">
                    <ModelSelector
                      modelId={validModelId}
                      onChange={handleModelChange}
                      predicting={predicting}
                      disabled={modelStatus !== 'ready'}
                    />

                    <div className="disclaimer">
                      <strong>Nota técnica:</strong> Los modelos usan calibración de temperatura
                      para ajustar las probabilidades de salida. El umbral de decisión es {UMBRAL}.
                      EfficientNetV2S ofrece el mejor equilibrio entre precisión y tamaño.
                    </div>
                  </div>
                </section>
              </div>
            </div>
          </ErrorBoundary>
        </main>

        <Footer hasExamplesDataset2={hasExamplesDataset2} />
      </div>
    </>
  );
}
