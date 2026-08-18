import { motion } from 'framer-motion';
import { MODELS } from '../lib/constants';

export default function ModelSelector({ modelId, onChange, predicting, disabled }) {
  return (
    <fieldset className="model-selector">
      <legend className="model-selector-title">Modelo de clasificación</legend>
      <div className="model-selector-options">
        {MODELS.map((m, i) => (
          <motion.label
            key={m.id}
            className={`model-card ${modelId === m.id ? 'is-active' : ''} ${m.auc == null ? 'is-pending' : ''}`}
            title={m.auc == null ? 'Modelo aún sin pesos publicados' : undefined}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.3 }}
            whileHover={m.auc != null ? { y: -2 } : {}}
            whileTap={m.auc != null ? { scale: 0.98 } : {}}
          >
            <input
              type="radio"
              name="modelId"
              value={m.id}
              checked={modelId === m.id}
              disabled={predicting || m.auc == null || disabled}
              onChange={() => {
                if (modelId !== m.id) onChange(m.id);
              }}
            />
            <div className="model-card-header">
              <span className="model-card-name">{m.name}</span>
              {m.auc != null && modelId === m.id && (
                <span className="model-card-check" aria-hidden="true">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M3 7l3 3 5-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
              )}
            </div>
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
            {modelId === m.id && disabled && (
              <span className="model-card-loading">cargando…</span>
            )}
          </motion.label>
        ))}
      </div>
    </fieldset>
  );
}
