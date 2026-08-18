import { motion, AnimatePresence } from 'framer-motion';

export default function ModelStatusBar({ status, progress, backend }) {
  return (
    <div className={`model-status status-${status}`} role="status" aria-live="polite">
      {status === 'loading' && (
        <>
          <span className="status-spinner" aria-hidden="true" />
          <span className="status-text">
            {progress >= 100
              ? 'Preparando el modelo… La primera vez puede tardar un poco.'
              : `Cargando pesos del modelo · ${progress}%`}
          </span>
          <span
            className="status-progress"
            style={{ width: `${progress}%` }}
            aria-hidden="true"
          />
        </>
      )}
      {status === 'ready' && (
        <>
          <span className="status-dot" aria-hidden="true" />
          <span>Modelo cargado · backend <code>{backend}</code></span>
        </>
      )}
      {status === 'error' && (
        <>
          <span className="status-dot status-dot--error" aria-hidden="true" />
          <span>No se pudo cargar el modelo</span>
          <button
            type="button"
            className="reanalyze-btn"
            onClick={() => window.location.reload()}
            style={{ marginTop: '0.5rem', width: 'auto', marginLeft: 'auto' }}
          >
            Reintentar
          </button>
        </>
      )}
    </div>
  );
}
