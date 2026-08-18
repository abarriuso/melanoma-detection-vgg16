import { motion } from 'framer-motion';

export default function Hero({ modelName, auc }) {
  return (
    <motion.header
      className="hero"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="hero-badge">
        <span className="hero-badge-dot" aria-hidden="true" />
        Inferencia 100% local
      </div>
      <h1>
        <span className="hero-title-gradient">Detección</span>
        <br />
        de melanoma
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
      <motion.p
        className="hero-warn"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
      >
        <span className="hero-warn-icon" aria-hidden="true">!</span>
        No es un dispositivo médico. Tasa de falsos negativos: ~12%.
        Si te preocupa una lesión, consulta a un dermatólogo.
      </motion.p>
    </motion.header>
  );
}
