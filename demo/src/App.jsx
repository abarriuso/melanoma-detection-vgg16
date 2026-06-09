import { useState, useEffect, useRef, useCallback, lazy, Suspense } from 'react';
import { loadModel, predictImage, getBackend } from './lib/model';
import { GITHUB_USER, REPO_NAME, DATASET_NAME, DATASET_URL, UMBRAL } from './lib/constants';
import { useCountUp } from './useCountUp';
import ErrorBoundary from './ErrorBoundary';
import './App.css';

// Grad-CAM se carga bajo demanda: solo si el usuario activa la atención.
// Mantiene el bundle del panel principal ligero (la dependencia interna
// reusa @tensorflow/tfjs que ya está cargado vía loadModel).
const loadGradCAM = () => import('./lib/gradcam');

// La galería de evaluación queda más abajo en la página y suele caer fuera
// del viewport inicial. Cargarla aparte mejora el TTFB del panel principal.
const ResultsGallery = lazy(() => import('./ResultsGallery'));

// Contador animado para el porcentaje de confianza.
function ConfidenceCounter({ confidence }) {
  const animated = useCountUp(confidence * 100, 600);
  return (
    <span className="result-conf" aria-live="polite">
      {animated.toFixed(1)}%
    </span>
  );
}

// Límite de tamaño de la imagen subida. 10 MB cubre cualquier dermatoscopia
// razonable; por encima suele ser un disparo accidental (PNG sin comprimir,
// captura de pantalla 4K, etc.) que solo va a saturar el decode.
const MAX_FILE_BYTES = 10 * 1024 * 1024;
// Solo formatos donde TF.js + canvas se comportan bien. WebP también funciona
// pero menos navegadores lo decodifican uniformemente; jpeg/png cubren el 99%.
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export default function App() {
  const [modelStatus, setModelStatus] = useState('loading'); // loading | ready | error
  const [progress, setProgress] = useState(0);
  const [backend, setBackend] = useState('—');

  const [imageURL, setImageURL] = useState(null);
  const [imageError, setImageError] = useState(false);
  // Mensaje de rechazo de archivo (tipo no permitido, tamaño excesivo, etc.).
  const [fileError, setFileError] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [autoRun, setAutoRun] = useState(false);
  const [examples, setExamples] = useState([]);

  const imgRef = useRef(null);
  const inputRef = useRef(null);
  const camCanvasRef = useRef(null);
  // Token de cancelación: cada análisis se asocia a un token incremental;
  // si llega un análisis nuevo, el resultado del viejo se descarta.
  const runTokenRef = useRef(0);
  const [showCam, setShowCam] = useState(false);
  const [camBusy, setCamBusy] = useState(false);
  // Flag de montaje compartido por efectos y análisis. Evita setState tras
  // unmount cuando una predicción (~100-300 ms en GPU) termina después de
  // que el usuario haya navegado fuera.
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);
  const base = import.meta.env.BASE_URL;

  // Ejemplos fijos del dataset (3 benignos + 3 malignos) para prueba rápida.
  // Elegidos del manifest como casos representativos y claros.
  // Además: si la URL trae ?sample=melanoma_X.jpg, intentamos precargar
  // y analizar esa muestra. Permite enlaces directos a casos concretos.
  useEffect(() => {
    let mounted = true;
    fetch(`${base}samples/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`manifest ${r.status}`);
        return r.json();
      })
      .then((d) => {
        if (!mounted) return;
        // 3 ejemplos de cada clase: los primeros del manifest (curados)
        const pickN = (arr, real, n = 3) =>
          arr.slice(0, n).map((name) => ({ real, path: `${base}samples/${real}/${name}` }));
        setExamples([...pickN(d.malignant, 'malignant'), ...pickN(d.benign, 'benign')]);

        // Resolución del parámetro ?sample=, con espera implícita a que el
        // modelo esté listo (autoRun arranca cuando la imagen decodifica).
        const params = new URLSearchParams(window.location.search);
        const wanted = params.get('sample');
        if (!wanted) return;
        const valid = /^melanoma_[\w-]+\.(jpe?g|png|webp)$/i.test(wanted);
        if (!valid) {
          setFileError('Formato de muestra inválido en la URL');
          return;
        }
        const real = d.malignant.includes(wanted)
          ? 'malignant'
          : d.benign.includes(wanted) ? 'benign' : null;
        if (!real) {
          setFileError(`Muestra "${wanted}" no encontrada en el conjunto de test`);
          return;
        }
        setImage(`${base}samples/${real}/${wanted}`, { auto: true });
      })
      .catch(() => mounted && setExamples([]));
    return () => {
      mounted = false;
    };
    // setImage cambia en cada render por las dependencias del callback;
    // omitirlo aquí evita reentrar al efecto y disparar fetch+setImage en
    // bucle. El efecto solo debe correr al montar (con `base` fijo).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  // Carga del modelo + metadatos técnicos. El flag `mounted` evita setState
  // tras desmontar el componente (React 18 StrictMode lo monta dos veces en dev).
  useEffect(() => {
    let mounted = true;
    loadModel((p) => mounted && setProgress(Math.round(p * 100)))
      .then(() => {
        if (!mounted) return;
        setBackend(getBackend());
        setModelStatus('ready');
      })
      .catch((err) => {
        console.error('Error cargando el modelo:', err);
        if (mounted) setModelStatus('error');
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Libera el objectURL (solo si es blob: de una subida) al cambiar/desmontar
  useEffect(() => {
    return () => {
      if (imageURL?.startsWith('blob:')) URL.revokeObjectURL(imageURL);
    };
  }, [imageURL]);

  const imagenLista = useCallback(
    () => !!imgRef.current && imgRef.current.complete && imgRef.current.naturalWidth > 0,
    [],
  );

  const analizar = useCallback(async () => {
    if (!imagenLista() || modelStatus !== 'ready') return;
    const myToken = ++runTokenRef.current;
    setPredicting(true);
    setPredictionError(null);
    try {
      const t0 = performance.now();
      const { raw, calibrated } = await predictImage(imgRef.current);
      const ms = Math.round(performance.now() - t0);
      // Si entretanto llegó otro análisis o el componente se desmontó, descartar.
      if (myToken !== runTokenRef.current || !mountedRef.current) return;
      const esMaligno = calibrated >= UMBRAL;
      // logit del score CRUDO del modelo (pre-calibración). Es la magnitud técnica.
      // `raw` viene ya clampado a [1e-7, 1-1e-7] desde model.js, así que el log es finito.
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
      if (mountedRef.current) setPredictionError('Error al analizar la imagen. Inténtalo de nuevo.');
    } finally {
      if (mountedRef.current && myToken === runTokenRef.current) setPredicting(false);
    }
  }, [modelStatus, imagenLista]);

  const setImage = useCallback(
    (url, { auto = false } = {}) => {
      setResult(null);
      setImageError(false);
      setShowCam(false);
      // El blob: anterior lo revoca el cleanup del useEffect que escucha imageURL.
      setImageURL((prev) => {
        // Si elegimos el MISMO ejemplo otra vez, la URL no cambia y onLoad no
        // se dispara. En ese caso, si la imagen ya está lista, lanzamos análisis.
        if (prev === url && auto && imagenLista()) {
          // Diferido para que setResult(null) tenga tiempo de aplicarse.
          queueMicrotask(() => analizar());
          return prev;
        }
        return url;
      });
      setAutoRun(auto);
    },
    [analizar, imagenLista],
  );

  const handleFile = (file) => {
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
    // Validar dimensiones mín/máx para evitar OOM con imágenes gigantes.
    const img = new Image();
    const objectURL = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(objectURL);
      const { naturalWidth: w, naturalHeight: h } = img;
      if (w < 16 || h < 16) {
        setFileError(`Imagen demasiado pequeña (${w}×${h}). Mínimo 16×16 píxeles.`);
        return;
      }
      if (w > 4096 || h > 4096) {
        setFileError(`Imagen demasiado grande (${w}×${h}). Máximo 4096×4096 píxeles.`);
        return;
      }
      setFileError(null);
      setImage(URL.createObjectURL(file));
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectURL);
      setFileError('No se pudo decodificar la imagen.');
    };
    img.src = objectURL;
  };

  const clearImage = () => {
    runTokenRef.current++; // invalida cualquier análisis en curso
    setImageURL(null);
    setImageError(false);
    setFileError(null);
    setResult(null);
    setPredicting(false);
    setShowCam(false);
  };

  // Calcula y pinta el Grad-CAM sobre el canvas overlay. La promesa
  // resulta en silencio si todo va bien; los errores se loguean pero no
  // rompen el resultado ya mostrado.
  const renderGradCAM = useCallback(async () => {
    if (!imgRef.current || !camCanvasRef.current) return;
    setCamBusy(true);
    try {
      const [{ computeGradCAM, paintHeatmap }, model] = await Promise.all([
        loadGradCAM(),
        loadModel(),
      ]);
      const heatmap = await computeGradCAM(model, imgRef.current);
      if (!mountedRef.current) return;
      paintHeatmap(camCanvasRef.current, heatmap);
    } catch (err) {
      console.error('Grad-CAM:', err);
      if (mountedRef.current) setShowCam(false);
    } finally {
      if (mountedRef.current) setCamBusy(false);
    }
  }, []);

  // Si el toggle se activa y ya hay un resultado pintado, generamos el
  // heatmap inmediatamente. Si se desactiva, limpiamos el canvas.
  useEffect(() => {
    if (!showCam || !result) return;
    renderGradCAM();
  }, [showCam, result, renderGradCAM]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    if (!dragActive) setDragActive(true);
  };

  const onPickFile = (e) => {
    handleFile(e.target.files?.[0]);
    e.target.value = '';
  };

  // Cuando la imagen termina de decodificar y venía de un ejemplo, analiza sola
  const onImgLoad = () => {
    if (autoRun) {
      setAutoRun(false);
      analizar();
    }
  };

  // Soporte de teclado para el dropzone (a11y)
  const onDropzoneKey = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  // Atajo global: Enter analiza si hay imagen lista y el modelo está
  // cargado. Se ignora si el foco está en un campo editable (no aplica
  // aquí porque no hay inputs de texto, pero conviene curarse en salud).
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Enter') return;
      const t = e.target;
      if (t?.matches?.('input, textarea, [contenteditable="true"], button')) return;
      if (!imageURL || imageError || modelStatus !== 'ready' || predicting) return;
      e.preventDefault();
      analizar();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [imageURL, imageError, modelStatus, predicting, analizar]);

  return (
    <div className="app" aria-busy={modelStatus === 'loading'}>
      <a href="#main-content" className="skip-link">Saltar al contenido principal</a>
      <nav className="toplinks" aria-label="Enlaces al código">
        <a
          href={`https://github.com/${GITHUB_USER}/${REPO_NAME}/blob/main/melanoma_detection_v2.ipynb`}
          target="_blank"
          rel="noreferrer"
        >
          Ver notebook →
        </a>
        <a
          href={`https://github.com/${GITHUB_USER}/${REPO_NAME}`}
          target="_blank"
          rel="noreferrer"
        >
          Código →
        </a>
      </nav>

      <header className="hero">
        <h1>Segunda opinión para lesiones de piel</h1>
        <p className="subtitle">
          Modelo VGG16 fine-tuned · AUC 0.9606 · ~0.2 s por análisis.
          Tus imágenes nunca salen de tu navegador.
        </p>
        <p className="hero-warn">
          No es un dispositivo médico. Tasa de falsos negativos: ~12%.
          Si te preocupa una lesión, consulta a un dermatólogo.
        </p>
      </header>

      <main id="main-content">
      <ErrorBoundary>
      <section className="panel" aria-labelledby="panel1-title">
        <div className="panel-head">
          <span className="panel-idx" aria-hidden="true">01</span>
          <h2 id="panel1-title">Sube una imagen</h2>
        </div>

        <div className="panel-body">
        <div className={`model-status status-${modelStatus}`} role="status" aria-live="polite">
          {modelStatus === 'loading' && (
            <>
              <span>Cargando pesos del modelo · {progress}%</span>
              <span
                className="status-progress"
                style={{ width: `${progress}%` }}
                aria-hidden="true"
              />
            </>
          )}
          {modelStatus === 'ready' && <>Modelo cargado · backend {backend}</>}
        {modelStatus === 'error' && (
          <>
            <p>No se pudo cargar el modelo</p>
            <button
              type="button"
              className="analyze-btn"
              onClick={() => window.location.reload()}
              style={{ marginTop: '0.5rem', fontSize: '0.82rem', padding: '0.5rem 1rem', width: 'auto' }}
            >
              Reintentar
            </button>
          </>
        )}
        </div>

        <p className="disclaimer disclaimer--top">
          Proyecto académico de investigación. No constituye un dispositivo médico
          ni sustituye la valoración de un profesional sanitario. El modelo tiene
          una tasa de falsos negativos del ~12%; consulta siempre a un dermatólogo.
          Diseñado exclusivamente para imágenes dermatoscópicas (capturadas con dermatoscopio).
          No uses fotos de teléfono móvil ni fotografías clínicas estándar.
        </p>

        <div
          className={`dropzone ${dragActive ? 'drag-active' : ''}`}
          role="button"
          tabIndex={0}
          aria-label="Seleccionar imagen de lesión a analizar"
          aria-describedby="dropzone-hint"
          aria-roledescription="zona de carga de imagen"
          onClick={() => inputRef.current?.click()}
          onKeyDown={onDropzoneKey}
          onDragOver={onDragOver}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(',')}
            capture="environment"
            hidden
            onChange={onPickFile}
          />
          {imageURL && !imageError ? (
            <div className="preview-wrap">
              <img
                ref={imgRef}
                src={imageURL}
                alt="Lesión dermatoscópica a analizar"
                className={`preview ${showCam ? 'is-grayscale' : ''}`}
                onLoad={onImgLoad}
                onError={() => setImageError(true)}
              />
              {predicting && <span className="scan-line" aria-hidden="true" />}
              <canvas
                ref={camCanvasRef}
                className={`preview-cam ${showCam ? 'is-on' : ''}`}
                aria-hidden="true"
              />
              <button
                type="button"
                className="preview-clear"
                onClick={(e) => { e.stopPropagation(); clearImage(); }}
                aria-label="Quitar imagen"
                title="Quitar imagen"
              >
                ×
              </button>
            </div>
          ) : (
            <div className="dropzone-hint">
              <span className="dropzone-icon" aria-hidden="true">+</span>
              {imageError ? (
                <>
                  <p>No se pudo abrir la imagen</p>
                  <p className="dropzone-sub">Prueba con otro archivo</p>
                </>
              ) : (
                <>
                  <p>Arrastra una imagen dermatoscópica</p>
                  <p className="dropzone-sub">o haz clic para seleccionar un archivo</p>
                </>
              )}
            </div>
          )}
        </div>

        {fileError && (
          <div className="file-error" role="alert">{fileError}</div>
        )}

        {predictionError && (
          <div className="file-error" role="alert">{predictionError}</div>
        )}

        {examples.length > 0 && (
          <div className="examples">
            <span className="examples-label">Ejemplos del conjunto de test</span>
            {/* aria-live para anunciar cambios en los ejemplos (ej. al recargar página) */}
            <div aria-live="polite" aria-atomic="true" className="sr-only" id="examples-announcer">
              {examples.length} ejemplos cargados: {examples.filter(e => e.real === 'malignant').length} malignos, {examples.filter(e => e.real === 'benign').length} benignos
            </div>
            <div className="examples-row" role="group" aria-label="Imágenes de ejemplo del dataset" aria-describedby="examples-announcer">
              {examples.map((ex) => (
                <button
                  key={ex.path}
                  type="button"
                  className={`example-thumb ${ex.real === 'malignant' ? 'is-mal' : 'is-ben'}`}
                  onClick={() => setImage(ex.path, { auto: true })}
                  disabled={modelStatus !== 'ready'}
                  aria-label={`Probar con lesión ${ex.real === 'malignant' ? 'maligna' : 'benigna'}`}
                  title={ex.real === 'malignant' ? 'Etiqueta: maligno' : 'Etiqueta: benigno'}
                >
                  <img
                    src={ex.path}
                    alt=""
                    loading="lazy"
                  />
                </button>
              ))}
            </div>
          </div>
        )}

        <button
          type="button"
          className="analyze-btn"
          onClick={analizar}
          disabled={!imageURL || imageError || modelStatus !== 'ready' || predicting}
          title="El análisis se ejecuta en tu navegador. La imagen no sale de tu dispositivo."
        >
          {predicting ? 'Analizando…' : 'Analizar imagen'}
        </button>

        {result && (
          <div className="cam-controls">
            <button
              type="button"
              className={`cam-btn ${showCam ? 'is-on' : ''}`}
              onClick={() => setShowCam((v) => !v)}
              disabled={camBusy}
              title="Visualiza qué regiones de la lesión influyeron más en la decisión del modelo (Grad-CAM). No indica dónde está el cáncer; el modelo puede equivocarse."
            >
              {camBusy
                ? 'Calculando relevancia…'
                : showCam ? 'Ocultar mapa de relevancia' : 'Mapa de relevancia (Grad-CAM)'}
            </button>
          </div>
        )}

        {result && (
          <div
            className={`result ${result.esMaligno ? 'malignant' : 'benign'}`}
            role="region"
            aria-label="Resultado del análisis"
          >
            <div className="result-top">
              <span className="result-label">{result.label}</span>
              <div className="result-meta">
                <span
                  className="result-latency"
                  title="Tiempo de inferencia en el navegador"
                >
                  {result.ms} ms
                </span>
                <ConfidenceCounter confidence={result.confidence} />
              </div>
            </div>
            <div
              className="confidence-bar"
              role="progressbar"
              aria-label="Confianza de la predicción"
              aria-valuenow={Math.round(result.confidence * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuetext={`${(result.confidence * 100).toFixed(1)}% — ${result.label}`}
            >
              <div
                className="confidence-fill"
                style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
              />
            </div>
            {!result.esMaligno && (
              <p className="result-reminder">
                Esto no es un diagnóstico. El modelo tiene ~12% de falsos negativos. Si tienes una lesión que te
                preocupa, consulta a un dermatólogo independientemente de esta herramienta.
              </p>
            )}
            {result.esMaligno && (
              <p className="result-reminder malignant-reminder">
                Esto no es un diagnóstico. Consulta a un dermatólogo para una evaluación clínica completa.
              </p>
            )}
            <p className="result-disclaimer">
              La confianza refleja la decisión del modelo, no el riesgo real. En clínica,
              la prevalencia de melanoma es muy baja (~1-5%). No considera tu historia clínica,
              exposición solar ni antecedentes familiares.
              <span className="threshold-note"> Umbral de decisión: {UMBRAL}.</span>
            </p>
          </div>
        )}

        </div>
      </section>
      </ErrorBoundary>

      <Suspense fallback={null}>
        <ErrorBoundary>
          <ResultsGallery />
        </ErrorBoundary>
      </Suspense>
      </main>

      <footer className="footer">
        <div className="footer-row">
          <a href={`https://github.com/${GITHUB_USER}/${REPO_NAME}`} target="_blank" rel="noreferrer">
            Código fuente
          </a>
          <span className="sep" aria-hidden="true">·</span>
          <a href={`https://github.com/${GITHUB_USER}`} target="_blank" rel="noreferrer">GitHub</a>
        </div>
        <div className="footer-dataset">
          Dataset:{' '}
          <a href={DATASET_URL} target="_blank" rel="noreferrer">{DATASET_NAME}</a>
          {' '}(CC0)
        </div>
      </footer>
    </div>
  );
}
