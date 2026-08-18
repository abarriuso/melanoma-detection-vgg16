import { motion } from 'framer-motion';

export default function GpuWarning() {
  return (
    <motion.div
      className="gpu-warning"
      role="status"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.3 }}
    >
      <svg className="gpu-warning-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="M8 1.5L15 13.5H1L8 1.5Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
        <path d="M8 6.2v3.3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        <circle cx="8" cy="11.6" r="0.7" fill="currentColor" />
      </svg>
      <div>
        <strong>La aceleración gráfica del navegador está desactivada o no disponible.</strong>{' '}
        La página funcionará, pero cada análisis puede tardar minutos en vez de segundos.
        En Chrome/Edge: Configuración → Sistema → «Usar aceleración por hardware cuando esté disponible», y recarga.
      </div>
    </motion.div>
  );
}
