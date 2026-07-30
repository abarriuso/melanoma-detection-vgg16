import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { loadModel, predictImage, getBackend, getGpuInfo, setActiveModelId } from './lib/model';
import { MODELS, getModel, GITHUB_USER, REPO_NAME, DATASET_NAME, DATASET_URL, EXAMPLES_DATASET2_NAME, EXAMPLES_DATASET2_URL, UMBRAL } from './lib/constants';
import { useCountUp } from './useCountUp';
import ErrorBoundary from './ErrorBoundary';
import './App.css';

// Grad-CAM se carga bajo demanda: solo si el usuario activa la atención.
// Mantiene el bundle del panel principal ligero (la dependencia interna
// reusa @tensorflow/tfjs que ya está cargado vía loadModel).
const BASE = import.meta.env.BASE_URL;
const loadGradCAM = () => import('./lib/gradcam');

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
// Tope de seguridad para la inferencia. En GPUs débiles o con el driver en
// mal estado, la primera compilación de shaders WebGL puede tardar mucho
// (medido: más de un minuto en una GT 710); pasado este tiempo asumimos que
// algo se ha atascado y lo comunicamos en vez de dejar el botón en
// "Analizando…" para siempre. Tras esa primera vez, la inferencia baja a
// segundos, así que reintentar suele funcionar.
const PREDICT_TIMEOUT_MS = 60_000;

function withTimeout(promise, ms, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms)),
  ]);
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function App() {
  const [modelStatus, setModelStatus] = useState('loading'); // loading | ready | error
  const [progress, setProgress] = useState(0);
  const [backend, setBackend] = useState('—');
  // true si el navegador no tiene aceleración gráfica utilizable
  const [slowGpu, setSlowGpu] = useState(false);

  // Con "reducir movimiento" activo en el sistema, las transiciones se
  // vuelven instantáneas (framer-motion ignora los variants vacíos).
  const prefersReducedMotion = useReducedMotion();
  const fadeUp = prefersReducedMotion
    ? {}
    : {
        initial: { opacity: 0, y: 8 },
        animate: { opacity: 1, y: 0 },
        exit: { opacity: 0, y: -4 },
        transition: { duration: 0.3, ease: 'easeOut' },
      };
  const fadeIn = prefersReducedMotion
    ? {}
    : {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        exit: { opacity: 0 },
        transition: { duration: 0.2 },
      };

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
  // Solo true si el manifest realmente trae imágenes isic17_* (es decir, si
  // ya se ejecutó scripts/download_examples_dataset.ps1 + refresh-samples.mjs).
  // No se declara como un hecho fijo: sería falso mientras no se haya hecho.
  const [hasExamplesDataset2, setHasExamplesDataset2] = useState(false);
  const [modelId, setModelId] = useState(() => {
    const stored = localStorage.getItem('modelId');
    const entry = stored && MODELS.find((m) => m.id === stored);
    // Ignora un modelId guardado de una sesión anterior si no existe o si
    // ese modelo todavía no tiene pesos publicados (evita quedar atascado
    // en el estado 'error' al recargar).
    return entry && entry.auc != null ? stored : 'vgg16';
  });

  const imgRef = useRef(null);
  const inputRef = useRef(null);
  const camCanvasRef = useRef(null);
  const runTokenRef = useRef(0);
  const [showCam, setShowCam] = useState(false);
  const [camBusy, setCamBusy] = useState(false);
  // Rect (en px, relativo al contenedor) donde la foto se ve de verdad
  // dentro de la caja de tamaño fijo del dropzone. La caja mide siempre lo
  // mismo (evita el salto de layout al cargar una imagen), pero la foto se
  // ajusta dentro con object-fit:contain y puede dejar bandas vacías a los
  // lados si no es cuadrada. Sin esto, el overlay de Grad-CAM se estira
  // sobre la caja entera en vez de sobre la foto, y no coincide con ella.
  const [camRect, setCamRect] = useState(null);
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);
  const pendingAutoRef = useRef(false);
  const analizarRef = useRef(null);
  const imageURLRef = useRef(null);
  const imageErrorRef = useRef(false);
  const predictingRef = useRef(false);
  imageURLRef.current = imageURL;
  imageErrorRef.current = imageError;
  predictingRef.current = predicting;

  // Ejemplos del dataset (3 benignos + 3 malignos) para prueba rápida. Al
  // cargar se muestran los primeros del manifest (curados); el botón
  // "rotar" pide 3+3 al azar del pool completo (que puede incluir un
  // segundo dataset, ver scripts/refresh-samples.mjs).
  // Además: si la URL trae ?sample=melanoma_X.jpg, intentamos precargar
  // y analizar esa muestra. Permite enlaces directos a casos concretos.
  const manifestRef = useRef(null);

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

  useEffect(() => {
    let mounted = true;
    fetch(`${BASE}samples/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`manifest ${r.status}`);
        return r.json();
      })
      .then((d) => {
        if (!mounted) return;
        manifestRef.current = d;
        setExamples(pickExamples(d));
        setHasExamplesDataset2(
          (d.malignant || []).some((f) => f.startsWith('isic17_')) ||
          (d.benign || []).some((f) => f.startsWith('isic17_')),
        );

        // Resolución del parámetro ?sample=, con cola si el modelo no está listo.
        const params = new URLSearchParams(window.location.search);
        const wanted = params.get('sample');
        if (!wanted) return;
        const valid = /^(melanoma|isic17)_[\w-]+\.(jpe?g|png|webp)$/i.test(wanted);
        if (!valid) {
          setFileError('Formato de muestra inválido en la URL');
          return;
        }
        if (!d.malignant || !d.benign) {
          setFileError('Manifest corrupto');
          return;
        }
        const real = d.malignant.includes(wanted)
          ? 'malignant'
          : d.benign.includes(wanted) ? 'benign' : null;
        if (!real) {
          setFileError(`Muestra "${wanted}" no encontrada en el conjunto de test`);
          return;
        }
        setImage(`${BASE}samples/${real}/${wanted}`, { auto: true });
      })
      .catch(() => mounted && setExamples([]));
    return () => {
      mounted = false;
    };
    // setImage cambia de identidad en cada render (depende de analizar);
    // este efecto solo debe correr una vez al montar para resolver ?sample=.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [BASE]);

  // Carga del modelo + metadatos técnicos. Se recarga si cambia modelId.
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
        // Sin aceleración por hardware (renderer software o backend CPU),
        // cada análisis pasa de décimas de segundo a minutos. Mejor
        // avisarlo que dejar que parezca que la página se ha colgado.
        const gpu = getGpuInfo();
        setSlowGpu(activeBackend === 'cpu' || !gpu.supported || gpu.software);
        setModelStatus('ready');
      })
      .catch((err) => {
        console.error('Error cargando el modelo:', err);
        if (mounted) setModelStatus('error');
      });
    return () => {
      mounted = false;
    };
  }, [modelId]);

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
      const { raw, calibrated } = await withTimeout(
        predictImage(imgRef.current, modelId),
        PREDICT_TIMEOUT_MS,
        'timeout',
      );
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
  }, [modelStatus, imagenLista, modelId]);
  analizarRef.current = analizar;

  // Retry auto-analysis when model finishes loading
  useEffect(() => {
    if (modelStatus === 'ready' && pendingAutoRef.current && imageURL) {
      pendingAutoRef.current = false;
      analizar();
    }
  }, [modelStatus, imageURL, analizar]);

  const setImage = useCallback(
    (url, { auto = false } = {}) => {
      setResult(null);
      setImageError(false);
      setShowCam(false);
      setCamRect(null); // se recalcula en onImgLoad; evita usar el de la foto anterior
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
      // auto:true: el análisis arranca solo al terminar de cargar la
      // imagen, igual que al elegir un ejemplo. No hace falta un botón.
      setImage(URL.createObjectURL(file), { auto: true });
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
    setCamRect(null);
  };

  // Calcula y pinta el Grad-CAM sobre el canvas overlay. La promesa
  // resulta en silencio si todo va bien; los errores se loguean pero no
  // rompen el resultado ya mostrado.
  const renderGradCAM = useCallback(async () => {
    if (!imgRef.current || !camCanvasRef.current) return;
    setCamBusy(true);
    setPredictionError(null);
    try {
      const [{ computeGradCAM, paintHeatmap }, model] = await Promise.all([
        loadGradCAM(),
        loadModel(),
      ]);
      const heatmap = await withTimeout(
        computeGradCAM(model, imgRef.current, modelId),
        PREDICT_TIMEOUT_MS,
        'timeout',
      );
      if (!mountedRef.current) return;
      paintHeatmap(camCanvasRef.current, heatmap, 224, 224);
    } catch (err) {
      console.error('Grad-CAM:', err);
      if (mountedRef.current) {
        setShowCam(false);
        // Antes el toggle se apagaba sin más y parecía que el botón no
        // hacía nada; mejor decir qué ha pasado.
        setPredictionError('No se pudo generar el mapa de relevancia. Inténtalo de nuevo.');
      }
    } finally {
      if (mountedRef.current) setCamBusy(false);
    }
  }, [modelId]);

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

  // Calcula dónde se ve la foto de verdad dentro de la caja de tamaño fijo
  // del dropzone (misma cuenta que hace CSS object-fit:contain, pero en JS
  // porque el overlay de Grad-CAM necesita esas coordenadas para dibujar
  // encima del sitio correcto, no de la caja entera).
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

  // Recalcula si cambia el ancho del contenedor (viewport responsive): la
  // altura del dropzone es fija por CSS, pero el ancho no.
  useEffect(() => {
    const img = imgRef.current;
    if (!img || !imageURL) return;
    const ro = new ResizeObserver(updateCamRect);
    ro.observe(img);
    return () => ro.disconnect();
  }, [imageURL, updateCamRect]);

  // Cuando la imagen termina de decodificar y venía de un ejemplo, analiza sola
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

  // Soporte de teclado para el dropzone (a11y)
  const onDropzoneKey = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  // Atajo global: Enter analiza si hay imagen lista y el modelo está
  // cargado. Usa refs para evitar re-registrar el listener en cada render.
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

  return (
    <div className="app" aria-busy={modelStatus === 'loading'}>
      <a href="#main-content" className="skip-link">Saltar al contenido principal</a>
      <nav className="toplinks" aria-label="Enlaces al código">
        <a
          href={`https://github.com/${GITHUB_USER}/${REPO_NAME}/blob/main/notebooks/vgg16.ipynb`}
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
        <h1>Detección de melanoma</h1>
        <p className="subtitle">
          Sube una foto dermatoscópica y la clasifica sola, sin pasos intermedios,
          como benigna o maligna. Por dentro hay una {getModel(modelId).name} —una
          red neuronal ya entrenada de fábrica con más de un millón de fotos— a la
          que le he hecho <em>fine-tuning</em> (un reentrenamiento especializado)
          con miles de imágenes de lesiones de piel, hasta un AUC de{' '}
          {getModel(modelId).auc ?? '—'} en test. Corre entera en tu navegador:
          la imagen nunca sale de tu dispositivo.
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
              {/* Con los pesos ya descargados queda el warmup: la primera
                  inferencia compila los shaders WebGL, y en GPUs modestas
                  eso puede llevar más de un minuto. Sin este mensaje, la
                  barra se queda clavada en 100% y parece que la app murió. */}
              <span>
                {progress >= 100
                  ? 'Preparando el modelo… La primera vez puede tardar un poco, sobre todo en equipos modestos.'
                  : `Cargando pesos del modelo · ${progress}%`}
              </span>
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
              className="reanalyze-btn"
              onClick={() => window.location.reload()}
              style={{ marginTop: '0.5rem', width: 'auto' }}
            >
              Reintentar
            </button>
          </>
        )}
        </div>

        <AnimatePresence>
        {slowGpu && modelStatus === 'ready' && (
          <motion.div className="gpu-warning" role="status" {...fadeUp}>
            <strong>La aceleración gráfica del navegador está desactivada o no
            disponible.</strong>{' '}
            La página funcionará, pero cada análisis puede tardar minutos en vez
            de segundos. En Chrome/Edge: Configuración → Sistema → «Usar
            aceleración por hardware cuando esté disponible», y recarga.
          </motion.div>
        )}
        </AnimatePresence>

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
          {/* Sin `capture`: ese atributo abre la cámara directamente en móvil,
              que es justo lo que el aviso de arriba pide no hacer (el modelo
              se entrenó con dermatoscopia, no con fotos de teléfono). */}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(',')}
            hidden
            onChange={onPickFile}
          />
          {imageURL && !imageError ? (
            <motion.div className="preview-wrap" {...fadeIn}>
              <img
                ref={imgRef}
                src={imageURL}
                alt="Lesión dermatoscópica a analizar"
                className={`preview ${showCam ? 'is-grayscale' : ''}`}
                onLoad={onImgLoad}
                onError={() => { if (mountedRef.current) setImageError(true); }}
              />
              {predicting && <span className="scan-line" aria-hidden="true" />}
              <canvas
                ref={camCanvasRef}
                className={`preview-cam ${showCam ? 'is-on' : ''}`}
                aria-hidden="true"
                style={camRect ? {
                  left: `${camRect.left}px`,
                  top: `${camRect.top}px`,
                  width: `${camRect.width}px`,
                  height: `${camRect.height}px`,
                } : undefined}
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
            </motion.div>
          ) : (
            <div className="dropzone-hint" id="dropzone-hint">
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

        {/* El análisis arranca solo al cargar la imagen (subida o ejemplo).
            Aquí solo queda el estado mientras corre y, si ya hay resultado
            o algo falló, un botón para repetirlo. Va justo debajo del
            dropzone y antes de "otros ejemplos" / "cambiar de modelo":
            son ajustes secundarios, no lo primero que hay que ver. */}
        {predicting && (
          <p className="analyzing-status" role="status" aria-live="polite">
            Analizando…
          </p>
        )}
        {!predicting && imageURL && !imageError && modelStatus === 'ready' && (result || predictionError) && (
          <button
            type="button"
            className="reanalyze-btn"
            onClick={analizar}
          >
            {predictionError ? 'Reintentar análisis' : 'Analizar de nuevo'}
          </button>
        )}

        <AnimatePresence>
        {result && (
          <motion.div className="cam-controls" {...fadeIn}>
            <button
              type="button"
              role="switch"
              aria-checked={showCam}
              className={`cam-toggle ${showCam ? 'is-on' : ''}`}
              onClick={() => setShowCam((v) => !v)}
              disabled={camBusy}
              title="Visualiza qué regiones de la lesión influyeron más en la decisión del modelo (Grad-CAM). No indica dónde está el cáncer; el modelo puede equivocarse."
            >
              <span className="cam-toggle-track" aria-hidden="true">
                <span className="cam-toggle-thumb" />
              </span>
              <span className="cam-toggle-label">
                {camBusy ? 'Calculando relevancia…' : 'Mapa de relevancia (Grad-CAM)'}
              </span>
            </button>
          </motion.div>
        )}
        </AnimatePresence>

        <AnimatePresence>
        {result && (
          <motion.div
            className={`result ${result.esMaligno ? 'malignant' : 'benign'}`}
            role="region"
            aria-label="Resultado del análisis"
            {...fadeUp}
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
          </motion.div>
        )}
        </AnimatePresence>

        {examples.length > 0 && (
          <div className="examples">
            <div className="examples-head">
              <span className="examples-label">Ejemplos del conjunto de test</span>
              <button
                type="button"
                className="rotate-examples-btn"
                onClick={rotateExamples}
                title="Cambiar por otras 6 muestras al azar"
              >
                <span aria-hidden="true">↻</span> Otros ejemplos
              </button>
            </div>
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
                    crossOrigin="anonymous"
                  />
                </button>
              ))}
            </div>
          </div>
        )}

        <fieldset className="model-selector">
          <legend className="model-selector-title">Modelo de clasificación</legend>
          <div className="model-selector-options">
            {MODELS.map((m) => (
              <label
                key={m.id}
                className={`model-card ${modelId === m.id ? 'is-active' : ''} ${m.auc == null ? 'is-pending' : ''}`}
                title={m.auc == null ? 'Modelo aún sin pesos publicados' : undefined}
              >
                <input
                  type="radio"
                  name="modelId"
                  value={m.id}
                  checked={modelId === m.id}
                  disabled={predicting || m.auc == null}
                  onChange={() => {
                    if (modelId !== m.id) {
                      setModelId(m.id);
                      localStorage.setItem('modelId', m.id);
                      clearImage();
                    }
                  }}
                />
                <span className="model-card-name">{m.name}</span>
                <span className="model-card-metrics">
                  {m.auc != null
                    ? `AUC ${m.auc} · ${m.sizeMB} MB`
                    : 'Pendiente de entrenamiento'}
                </span>
                {m.auc != null && (
                  <span className="model-card-detail">
                    Sens {m.sens} · Esp {m.spec}
                  </span>
                )}
                {modelId === m.id && modelStatus === 'loading' && (
                  <span className="model-card-loading">cargando…</span>
                )}
              </label>
            ))}
          </div>
        </fieldset>

        </div>
      </section>
      </ErrorBoundary>
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
          {hasExamplesDataset2 ? 'Dataset de entrenamiento' : 'Dataset'}:{' '}
          <a href={DATASET_URL} target="_blank" rel="noreferrer">{DATASET_NAME}</a>
          {' '}(CC0)
        </div>
        {hasExamplesDataset2 && (
          <div className="footer-dataset">
            Ejemplos también de:{' '}
            <a href={EXAMPLES_DATASET2_URL} target="_blank" rel="noreferrer">{EXAMPLES_DATASET2_NAME}</a>
            {' '}(CC0)
          </div>
        )}
      </footer>
    </div>
  );
}
