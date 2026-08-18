import { motion, AnimatePresence } from 'framer-motion';

export default function GradCamToggle({ showCam, onToggle, busy, disabled }) {
  return (
    <AnimatePresence>
      <motion.div
        className="cam-controls"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <button
          type="button"
          role="switch"
          aria-checked={showCam}
          className={`cam-toggle ${showCam ? 'is-on' : ''}`}
          onClick={() => onToggle()}
          disabled={disabled || busy}
          title="Visualiza qué regiones de la lesión influyeron más en la decisión del modelo (Grad-CAM). No indica dónde está el cáncer; el modelo puede equivocarse."
        >
          <span className="cam-toggle-track" aria-hidden="true">
            <motion.span
              className="cam-toggle-thumb"
              animate={{ x: showCam ? 18 : 0 }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            />
          </span>
          <span className="cam-toggle-label">
            {busy ? 'Calculando relevancia…' : 'Mapa de relevancia (Grad-CAM)'}
          </span>
        </button>
      </motion.div>
    </AnimatePresence>
  );
}
