import { motion } from 'framer-motion';
import ConfidenceGauge from './ConfidenceGauge';

export default function ResultCard({ result }) {
  if (!result) return null;

  const { label, confidence, ms, esMaligno } = result;
  const statusColor = esMaligno ? 'var(--malignant)' : 'var(--benign)';
  const statusBg = esMaligno ? 'var(--malignant-bg)' : 'var(--benign-bg)';
  const statusBorder = esMaligno ? 'var(--malignant-border)' : 'var(--benign-border)';
  const statusGlow = esMaligno ? 'var(--malignant-glow)' : 'var(--benign-glow)';
  const statusBright = esMaligno ? 'var(--malignant-bright)' : 'var(--benign-bright)';

  return (
    <motion.div
      className="result-card"
      role="region"
      aria-label="Resultado del análisis"
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -8, scale: 0.98 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      style={{
        background: statusBg,
        borderColor: statusBorder,
        boxShadow: `0 0 30px ${statusGlow}`,
      }}
    >
      <div className="result-card-header">
        <div className="result-card-label-group">
          <span
            className="result-card-status-dot"
            style={{ background: statusColor, boxShadow: `0 0 8px ${statusGlow}` }}
            aria-hidden="true"
          />
          <span className="result-card-label" style={{ color: statusBright }}>
            {label}
          </span>
        </div>
        <span
          className="result-card-latency"
          title="Tiempo de inferencia en el navegador"
        >
          {ms} ms
        </span>
      </div>

      <div className="result-card-body">
        <ConfidenceGauge confidence={confidence} isMalignant={esMaligno} size={140} />
        <div className="result-card-details">
          <div className="result-detail-row">
            <span className="result-detail-key">Probabilidad calibrada</span>
            <span className="result-detail-value" style={{ color: statusBright }}>
              {(result.score * 100).toFixed(2)}%
            </span>
          </div>
          <div className="result-detail-row">
            <span className="result-detail-key">Logit</span>
            <span className="result-detail-value mono">{result.logit.toFixed(3)}</span>
          </div>
          <div className="result-detail-row">
            <span className="result-detail-key">Umbral</span>
            <span className="result-detail-value mono">0.5</span>
          </div>
        </div>
      </div>

      <div className="result-card-footer">
        <p className="result-reminder">
          {esMaligno
            ? 'Esto no es un diagnóstico. Consulta a un dermatólogo para una evaluación clínica completa.'
            : 'Esto no es un diagnóstico. El modelo tiene ~12% de falsos negativos. Si tienes una lesión que te preocupa, consulta a un dermatólogo.'}
        </p>
        <p className="result-disclaimer">
          La confianza refleja la decisión del modelo, no el riesgo real.
          En clínica, la prevalencia de melanoma es muy baja (~1-5%).
          No considera historia clínica, exposición solar ni antecedentes familiares.
        </p>
      </div>
    </motion.div>
  );
}
