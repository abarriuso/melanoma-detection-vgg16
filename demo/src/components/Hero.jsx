import { motion } from 'framer-motion';

export default function Hero({ modelName, auc }) {
  return (
    <motion.header
      className="hero"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      <p className="hero-eyebrow">Clasificación de lesiones dermatoscópicas · Inferencia 100 % local</p>
      <h1>
        Detección de <em>melanoma</em>
      </h1>
      <p className="subtitle">
        Sube una foto dermatoscópica y la clasifica sola, sin pasos intermedios,
        como benigna o maligna. Por dentro hay una {modelName} —una red neuronal
        ya entrenada de fábrica con más de un millón de fotos— a la que le hemos
        hecho <em>fine-tuning</em> con miles de imágenes de lesiones de piel,
        hasta un AUC de{' '}
        <span className="metric-highlight">{auc ?? '—'}</span> en test.
        Corre entera en tu navegador: la imagen nunca sale de tu dispositivo.
      </p>
      <p className="hero-warn">
        <svg className="hero-warn-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M8 1.5L15 13.5H1L8 1.5Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
          <path d="M8 6.2v3.3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          <circle cx="8" cy="11.6" r="0.7" fill="currentColor" />
        </svg>
        <span>
          No es un dispositivo médico. Tasa de falsos negativos: ~12 %.
          Si te preocupa una lesión, consulta a un dermatólogo.
        </span>
      </p>
    </motion.header>
  );
}
