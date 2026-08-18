import { motion, AnimatePresence } from 'framer-motion';

export default function ExampleGallery({ examples, onSelect, disabled }) {
  if (examples.length === 0) return null;

  return (
    <div className="examples">
      <div className="examples-head">
        <span className="examples-label">Ejemplos del conjunto de test</span>
        <button
          type="button"
          className="rotate-examples-btn"
          onClick={() => onSelect('rotate')}
          title="Cambiar por otras 6 muestras al azar"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M1 7a6 6 0 016-6m0 0l-2 2m2-2l2 2M13 7a6 6 0 01-6 6m0 0l2-2m-2 2l-2-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Otros ejemplos
        </button>
      </div>
      <div aria-live="polite" aria-atomic="true" className="sr-only" id="examples-announcer">
        {examples.length} ejemplos cargados: {examples.filter(e => e.real === 'malignant').length} malignos, {examples.filter(e => e.real === 'benign').length} benignos
      </div>
      <div className="examples-row" role="group" aria-label="Imágenes de ejemplo del dataset" aria-describedby="examples-announcer">
        <AnimatePresence>
          {examples.map((ex) => (
            <motion.button
              key={ex.path}
              type="button"
              className={`example-thumb ${ex.real === 'malignant' ? 'is-mal' : 'is-ben'}`}
              onClick={() => onSelect(ex.path)}
              disabled={disabled}
              aria-label={`Probar con lesión ${ex.real === 'malignant' ? 'maligna' : 'benigna'}`}
              title={ex.real === 'malignant' ? 'Etiqueta: maligno' : 'Etiqueta: benigno'}
              layout
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              transition={{ duration: 0.2 }}
            >
              <img src={ex.path} alt="" loading="lazy" crossOrigin="anonymous" />
              <span className="example-thumb-badge" aria-hidden="true">
                {ex.real === 'malignant' ? 'M' : 'B'}
              </span>
            </motion.button>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
