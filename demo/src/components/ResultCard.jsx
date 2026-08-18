import { motion } from 'framer-motion';
import ConfidenceGauge from './ConfidenceGauge';

const rowVariants = {
  hidden: { opacity: 0, y: 6 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.15 + i * 0.07, duration: 0.3, ease: [0.22, 1, 0.36, 1] },
  }),
};

export default function ResultCard({ result }) {
  if (!result) return null;

  const { label, confidence, ms, esMaligno } = result;
  const statusColor = esMaligno ? 'var(--malignant)' : 'var(--benign)';

  return (
    <motion.div
      className="result-card"
      role="region"
      aria-label="Resultado del análisis"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      <p className="result-card-eyebrow">Lectura del instrumento</p>

      <div className="result-card-header">
        <div className="result-card-label-group">
          <span
            className="result-card-status-dot"
            style={{ background: statusColor }}
            aria-hidden="true"
          />
          <span className="result-card-label" style={{ color: statusColor }}>
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
          <motion.div
            className="result-detail-row"
            custom={0}
            variants={rowVariants}
            initial="hidden"
            animate="visible"
          >
            <span className="result-detail-key">Probabilidad calibrada</span>
            <span className="result-detail-value" style={{ color: statusColor }}>
              {(result.score * 100).toFixed(2)}%
            </span>
          </motion.div>
          <motion.div
            className="result-detail-row"
            custom={1}
            variants={rowVariants}
            initial="hidden"
            animate="visible"
          >
            <span className="result-detail-key">Logit</span>
            <span className="result-detail-value mono">{result.logit.toFixed(3)}</span>
          </motion.div>
          <motion.div
            className="result-detail-row"
            custom={2}
            variants={rowVariants}
            initial="hidden"
            animate="visible"
          >
            <span className="result-detail-key">Umbral</span>
            <span className="result-detail-value mono">0.5</span>
          </motion.div>
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
