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
      <span className="gpu-warning-icon" aria-hidden="true">⚡</span>
      <div>
        <strong>La aceleración gráfica del navegador está desactivada o no disponible.</strong>{' '}
        La página funcionará, pero cada análisis puede tardar minutos en vez de segundos.
        En Chrome/Edge: Configuración → Sistema → «Usar aceleración por hardware cuando esté disponible», y recarga.
      </div>
    </motion.div>
  );
}
