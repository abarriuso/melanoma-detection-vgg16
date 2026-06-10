import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { predictImage } from './lib/model';
import { DATASET_NAME, DATASET_URL } from './lib/constants';
import './ResultsGallery.css';

// Versión del cache de scores. Se incluye modelId en la clave para que
// cada modelo tenga su propio cache (modelos distintos dan scores distintos).
const SCORES_VERSION = 1;
const SCORES_KEY = `samples-scores-v${SCORES_VERSION}`;
// Tamaño del lote visible. El pool real es mucho mayor (120+); en pantalla
// solo mostramos un subconjunto manejable.
const BATCH_SIZE = 30;

const MODES = [
  { id: 'random', label: 'Aleatorio', desc: 'Selección aleatoria del pool de test.' },
  { id: 'hardest', label: 'Más dudosos', desc: 'Las muestras con score más cercano al umbral (0.5).' },
  { id: 'fn', label: 'Falsos negativos', desc: 'Malignos que el modelo etiqueta como benignos. El error clínico crítico.' },
  { id: 'fp', label: 'Falsos positivos', desc: 'Benignos que el modelo etiqueta como malignos. Biopsia innecesaria.' },
];

export function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Lee el cache de scores de localStorage. Tolerante: si el JSON está
// corrupto o la versión no coincide, devolvemos {} sin reventar.
export function loadScoresCache(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return (parsed && typeof parsed === 'object' ? parsed : {});
  } catch {
    return {};
  }
}

export function saveScoresCache(map, key) {
  try {
    localStorage.setItem(key, JSON.stringify(map));
    return true;
  } catch {
    // localStorage lleno o deshabilitado; degradamos en silencio.
    return false;
  }
}

// Verifica si localStorage está disponible
export function isLocalStorageAvailable() {
  try {
    const test = '__storage_test__';
    localStorage.setItem(test, test);
    localStorage.removeItem(test);
    return true;
  } catch {
    return false;
  }
}

// Carga una imagen oculta y la decodifica. Necesaria para clasificar el
// pool entero en background sin pintarlas todas en el DOM.
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`load ${src}`));
    img.src = src;
  });
}

export default function ResultsGallery({ modelId }) {
  const base = import.meta.env.BASE_URL;
  const SCORES_KEY = `samples-scores-v${SCORES_VERSION}-${modelId}`;

  const [pool, setPool] = useState([]);          // flat: [{file, real, path}] x ~120
  const [mode, setMode] = useState('random');
  const [samples, setSamples] = useState([]);     // visible: [{...pool, pred, score}]
  const [scoresMap, setScoresMap] = useState(() => loadScoresCache(SCORES_KEY));
  const [scoringStatus, setScoringStatus] = useState('idle'); // idle | scoring | error
  const [scoringProgress, setScoringProgress] = useState({ done: 0, total: 0 });
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [storageAvailable, setStorageAvailable] = useState(() => isLocalStorageAvailable());

  const imgRefs = useRef({});
  // Token de cancelación de la operación pesada (scoring full pool); si el
  // usuario navega o cambia de modo a media clasificación, abortamos.
  const opTokenRef = useRef(0);
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  // Limpia referencias de imágenes al cambiar de modo o desmontar para evitar memory leaks
  useEffect(() => {
    return () => {
      imgRefs.current = {};
    };
  }, [mode]);

  // ¿Tenemos score para cada muestra del pool?
  const allScored = useMemo(() => {
    if (pool.length === 0) return false;
    return pool.every((s) => scoresMap[s.file] != null);
  }, [pool, scoresMap]);

  // Construye un sample con su score/pred derivado, si está en el cache.
  const decorate = useCallback(
    (s) => {
      const score = scoresMap[s.file];
      if (score == null) return { ...s, pred: null, score: null };
      return { ...s, score, pred: score >= 0.5 ? 'malignant' : 'benign' };
    },
    [scoresMap],
  );

  // Construye el batch visible según el modo activo. Asume que para los
  // modos != 'random' los scores ya están disponibles.
  const buildSamples = useCallback(
    (m) => {
      if (pool.length === 0) return [];
      if (m === 'random') {
        return shuffle(pool).slice(0, BATCH_SIZE).map((s) => decorate(s));
      }
      const decorated = pool.map((s) => decorate(s)).filter((s) => s.score != null);
      if (m === 'hardest') {
        const deltas = decorated.map((s) => ({ s, d: Math.abs(s.score - 0.5) }));
        deltas.sort((a, b) => a.d - b.d);
        return deltas.slice(0, BATCH_SIZE).map(({ s }) => s);
      }
      if (m === 'fn') {
        return decorated
          .filter((s) => s.real === 'malignant' && s.pred === 'benign')
          .sort((a, b) => a.score - b.score)
          .slice(0, BATCH_SIZE);
      }
      if (m === 'fp') {
        return decorated
          .filter((s) => s.real === 'benign' && s.pred === 'malignant')
          .sort((a, b) => b.score - a.score)
          .slice(0, BATCH_SIZE);
      }
      return [];
    },
    [pool, decorate],
  );

  // Carga el manifest y prepara el pool.
  useEffect(() => {
    let alive = true;
    fetch(`${base}samples/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`manifest ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (!alive) return;
        const flat = [
          ...data.benign.map((f) => ({ file: f, real: 'benign', path: `${base}samples/benign/${f}` })),
          ...data.malignant.map((f) => ({ file: f, real: 'malignant', path: `${base}samples/malignant/${f}` })),
        ];
        setPool(flat);
      })
      .catch(() => alive && setPool([]));
    return () => { alive = false; };
  }, [base]);

  // Primera vez que tenemos pool, sembramos el batch aleatorio.
  useEffect(() => {
    if (pool.length > 0 && samples.length === 0) {
      setSamples(buildSamples('random'));
    }
    // samples.length controla la siembra inicial; no queremos que cambios
    // posteriores en samples vuelvan a disparar este efecto.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pool]);

  // Reaccionamos a cambios de modo cuando los scores ya están listos.
  // Si no lo están, ensureScored() los pedirá y volverá a llamar a este
  // efecto cuando lleguen vía scoresMap.
  useEffect(() => {
    if (pool.length === 0) return;
    if (mode === 'random') return; // 'random' no necesita scores
    if (!allScored) return;
    setSamples(buildSamples(mode));
    setDone(true);
  }, [mode, allScored, pool, buildSamples]);

  // Clasifica en background todas las muestras del pool que aún no estén
  // en scoresMap. Pinta el resultado en el cache de forma incremental.
  const ensureScored = useCallback(async () => {
    if (allScored) return;
    const myToken = ++opTokenRef.current;
    setScoringStatus('scoring');
    const missing = pool.filter((s) => scoresMap[s.file] == null);
    setScoringProgress({ done: 0, total: missing.length });
    const next = { ...scoresMap };
    for (let i = 0; i < missing.length; i++) {
      if (myToken !== opTokenRef.current) return; // cancelado
      const s = missing[i];
      try {
        const img = await loadImage(s.path);
        const { calibrated } = await predictImage(img, modelId);
        next[s.file] = calibrated;
        if (i % 5 === 0 || i === missing.length - 1) {
          if (!mountedRef.current) return;
          setScoresMap({ ...next });
          const saved = saveScoresCache(next, SCORES_KEY);
          if (!saved && storageAvailable) {
            setStorageAvailable(false);
          }
        }
        setScoringProgress({ done: i + 1, total: missing.length });
      } catch (err) {
        console.error('scoring', s.file, err);
        if (mountedRef.current && myToken === opTokenRef.current) {
          setScoringStatus('error');
        }
        // No abortamos el pool por un fallo individual
      }
    }
    if (!mountedRef.current || myToken !== opTokenRef.current) return;
    setScoresMap(next);
    const saved = saveScoresCache(next, SCORES_KEY);
    if (!saved && storageAvailable) {
      setStorageAvailable(false);
    }
    setScoringStatus('idle');
  }, [pool, scoresMap, allScored, storageAvailable]);

  // Si el usuario elige un modo que necesita scores y no los hay, los pedimos.
  useEffect(() => {
    if (mode === 'random') return;
    if (allScored) return;
    if (scoringStatus === 'scoring') return;
    ensureScored();
  }, [mode, allScored, scoringStatus, ensureScored]);

  // En modo random + sin desafío, "Clasificar todas" hace lo de siempre.
  // Actualiza el estado cada 5 predicciones en vez de cada una para
  // evitar ~60 re-renders innecesarios con 30 items.
  // Usa functional setState para scoresMap para evitar stale closure.
  const clasificarTodas = async () => {
    const myToken = ++opTokenRef.current;
    const scoreUpdates = {};
    setRunning(true);
    try {
      const updated = [...samples];
      for (let i = 0; i < updated.length; i++) {
        if (myToken !== opTokenRef.current || !mountedRef.current) return;
        const img = imgRefs.current[i];
        if (!img) continue;
        try {
          if (!(img.complete && img.naturalWidth > 0)) await img.decode();
          const { calibrated } = await predictImage(img, modelId);
          updated[i] = {
            ...updated[i],
            score: calibrated,
            pred: calibrated >= 0.5 ? 'malignant' : 'benign',
          };
          if (scoresMap[updated[i].file] == null) {
            scoreUpdates[updated[i].file] = calibrated;
          }
          if (i % 5 === 0 || i === updated.length - 1) {
            if (!mountedRef.current || myToken !== opTokenRef.current) return;
            setSamples([...updated]);
          }
        } catch (err) {
          console.error('clasificar', i, err);
        }
      }
      if (Object.keys(scoreUpdates).length > 0) {
        setScoresMap((prev) => {
          const merged = { ...prev, ...scoreUpdates };
          const saved = saveScoresCache(merged, SCORES_KEY);
          if (!saved && storageAvailable) {
            setStorageAvailable(false);
          }
          return merged;
        });
      }
      if (!mountedRef.current || myToken !== opTokenRef.current) return;
      setSamples([...updated]);
      setDone(true);
    } finally {
      if (mountedRef.current && myToken === opTokenRef.current) setRunning(false);
    }
  };

  // Reordenar: en modo random saca un sample nuevo; en los otros modos
  // simplemente vuelve a aplicar el modo (que ya está deterministicamente
  // ordenado, pero limpia preds visuales si el usuario está jugando challenge).
  const reordenar = () => {
    imgRefs.current = {};
    setDone(false);
    setSamples(buildSamples(mode));
  };

  const onModeChange = (next) => {
    if (next === mode) return;
    setMode(next);
    setDone(false);
    if (next === 'random') {
      setSamples(buildSamples('random'));
    }
    // Para los otros modos, el efecto que vigila [mode, allScored] se
    // encarga de reconstruir samples cuando haya scores.
  };

  // Stats sobre las samples actualmente clasificadas.
  const stats = useMemo(() => {
    const classified = samples.filter((s) => s.pred != null);
    if (classified.length === 0) return null;
    const correct = classified.filter((s) => s.pred === s.real).length;
    const fn = classified.filter((s) => s.real === 'malignant' && s.pred === 'benign').length;
    const fp = classified.filter((s) => s.real === 'benign' && s.pred === 'malignant').length;
    return {
      total: classified.length,
      correct,
      accuracy: Math.round((correct / classified.length) * 100),
      fn,
      fp,
    };
  }, [samples]);

  // Handlers de tilt 3D para las cards. Estables (solo usan e.currentTarget).
  const onCardMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    e.currentTarget.style.transform =
      `perspective(600px) rotateY(${x * 12}deg) rotateX(${-y * 12}deg) scale(1.04)`;
  }, []);

  const onCardLeave = useCallback((e) => {
    e.currentTarget.style.transform = '';
  }, []);

  const setImgRef = useCallback((el, i) => {
    if (el) imgRefs.current[i] = el;
    else delete imgRefs.current[i];
  }, []);

  if (pool.length === 0) return null;

  const currentMode = MODES.find((m) => m.id === mode) || MODES[0];
  const noResults = mode !== 'random' && allScored && samples.length === 0;
  const showClassifyButton = mode === 'random';

  return (
    <section className="panel results-section" aria-labelledby="panel2-title">
      <div className="panel-head">
        <span className="panel-idx" aria-hidden="true">02</span>
        <h2 id="panel2-title">Evaluación sobre el conjunto de test</h2>
      </div>
      <div className="panel-body">
      <p className="results-intro">
        Pool de {pool.length} imágenes del conjunto de test del{' '}
        <a href={DATASET_URL} target="_blank" rel="noreferrer">{DATASET_NAME}</a>{' '}
        (Kaggle, CC0). El modelo las procesa en el navegador y el resultado se
        contrasta con la etiqueta real. Dataset de balance artificial (50/50);
        la prevalencia clínica real es ≪50%.
      </p>

      <div className="mode-selector" role="tablist" aria-label="Modo de selección">
        {MODES.map((m, idx) => (
          <button
            key={m.id}
            type="button"
            role="tab"
            id={`tab-${m.id}`}
            aria-controls={`tabpanel-${m.id}`}
            aria-selected={mode === m.id}
            className={`mode-btn ${mode === m.id ? 'is-active' : ''}`}
            onClick={() => onModeChange(m.id)}
            onKeyDown={(e) => {
              const tabs = MODES;
              let next = -1;
              if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                next = (idx + 1) % tabs.length;
              } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                next = (idx - 1 + tabs.length) % tabs.length;
              }
              if (next >= 0) {
                onModeChange(tabs[next].id);
                e.currentTarget.parentElement.children[next].focus();
              }
            }}
            disabled={running}
            title={m.desc}
          >
            {m.label}
          </button>
        ))}
      </div>
      <p className="mode-desc">{currentMode.desc}</p>

      <div
        role="tabpanel"
        id={`tabpanel-${mode}`}
        aria-labelledby={`tab-${mode}`}
      >

      <div className="batch-actions">
        {showClassifyButton && (
          <button
            type="button"
            className="batch-btn"
            onClick={clasificarTodas}
            disabled={running || done}
          >
            {running ? 'Clasificando…' : done ? 'Completado' : 'Clasificar todas'}
          </button>
        )}
        <button
          type="button"
          className="batch-btn batch-btn-secondary"
          onClick={reordenar}
          disabled={running || scoringStatus === 'scoring'}
        >
          Reordenar
        </button>
      </div>

      {scoringStatus === 'scoring' && (
        <div className="scoring-banner" role="status" aria-live="polite">
          <span>
            Evaluando el pool del test ({scoringProgress.done}/{scoringProgress.total})
          </span>
          <div className="scoring-bar">
            <div
              className="scoring-fill"
              style={{
                width: scoringProgress.total
                  ? `${(scoringProgress.done / scoringProgress.total) * 100}%`
                  : '0%',
              }}
            />
          </div>
        </div>
      )}

      {scoringStatus === 'error' && (
        <div className="scoring-banner scoring-error" role="alert" aria-live="assertive">
          <span>Error al evaluar el pool. Algunos scores no pudieron calcularse.</span>
          <button
            type="button"
            className="batch-btn batch-btn-secondary"
            onClick={() => {
              setScoringStatus('idle');
              ensureScored();
            }}
          >
            Reintentar
          </button>
        </div>
      )}

      {!storageAvailable && (
        <div className="scoring-banner storage-warning" role="status" aria-live="polite">
          <span>
            <strong>Almacenamiento local no disponible</strong> (modo privado o cuota agotada).
            Los scores no se guardarán entre sesiones.
          </span>
        </div>
      )}

      {noResults && (
        <div className="empty-state" role="status">
          No hay muestras para este modo en el pool actual.
        </div>
      )}

      {stats && (
        <div className="batch-stats" role="status" aria-live="polite">
          <div className="stat">
            <span className="stat-value">{stats.accuracy}%</span>
            <span className="stat-label">Exactitud</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stats.correct}/{stats.total}</span>
            <span className="stat-label">Aciertos</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stats.fn}</span>
            <span className="stat-label">Falsos neg.</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stats.fp}</span>
            <span className="stat-label">Falsos pos.</span>
          </div>
        </div>
      )}

      <div className="samples-grid" aria-label="Muestras del conjunto de test">
        {samples.map((sample, i) => {
          const revealed = sample.pred != null;
          const correctModel = revealed && sample.pred === sample.real;
          let borderClass = '';
          if (revealed) borderClass = correctModel ? 'card-correct' : 'card-wrong';

          const realLetter = sample.real === 'malignant' ? 'M' : 'B';

          return (
            <div
              key={`${mode}-${i}-${sample.file}`}
              className={`sample-card ${borderClass}`}
              style={{ animationDelay: `${i * 40}ms` }}
              onMouseMove={onCardMove}
              onMouseLeave={onCardLeave}
            >
              <span
                className={`sample-real-badge real-${sample.real}`}
                aria-hidden="true"
                title={sample.real === 'malignant' ? 'Real: maligno' : 'Real: benigno'}
              >
                {realLetter}
              </span>
              <img
                ref={(el) => setImgRef(el, i)}
                src={sample.path}
                alt={`Lesión dermatoscópica ${sample.real === 'malignant' ? 'maligna' : 'benigna'}`}
                loading="lazy"
                decoding="async"
              />
              <div className="sample-info">
                <span className="sample-real">
                  Etiqueta: {sample.real === 'malignant' ? 'Maligno' : 'Benigno'}
                </span>
                {revealed && (
                  <span className={`sample-pred ${correctModel ? 'pred-ok' : 'pred-fail'}`}>
                    Modelo: {sample.pred === 'malignant' ? 'Maligno' : 'Benigno'} · {(sample.score * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
      </div>
      </div>
    </section>
  );
}
