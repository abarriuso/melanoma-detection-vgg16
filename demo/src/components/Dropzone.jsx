import { motion, AnimatePresence } from 'framer-motion';

export default function Dropzone({
  imageURL,
  imageError,
  dragActive,
  predicting,
  showCam,
  camRect,
  onFile,
  onDrop,
  onDragOver,
  onDragLeave,
  onClear,
  onImageLoad,
  onImageError,
  imgRef,
  camCanvasRef,
  inputRef,
  disabled,
}) {
  const onDropzoneKey = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  const onPickFile = (e) => {
    onFile(e.target.files?.[0]);
    e.target.value = '';
  };

  return (
    <div className="dropzone-wrapper">
      <div
        className={`dropzone ${dragActive ? 'drag-active' : ''} ${imageURL ? 'has-image' : ''} ${predicting ? 'is-scanning' : ''}`}
        role="button"
        tabIndex={0}
        aria-label="Seleccionar imagen de lesión a analizar"
        aria-describedby="dropzone-hint"
        aria-roledescription="zona de carga de imagen"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={onDropzoneKey}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          hidden
          onChange={onPickFile}
        />
        <AnimatePresence mode="wait">
          {imageURL && !imageError ? (
            <motion.div
              key="preview"
              className="preview-stage"
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Retícula de registro: anillo fino + marcas cardinales,
                  como una preparación de microscopio. */}
              <svg className="preview-reticle" viewBox="-6 -6 112 112" aria-hidden="true">
                <circle cx="50" cy="50" r="52.5" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <path
                  d="M50 -5v6M50 99v6M-5 50h6M99 50h6"
                  stroke="currentColor"
                  strokeWidth="0.9"
                />
              </svg>
              <div className="preview-wrap">
                <img
                  ref={imgRef}
                  src={imageURL}
                  alt="Lesión dermatoscópica a analizar"
                  className={`preview ${showCam ? 'is-grayscale' : ''}`}
                  onLoad={onImageLoad}
                  onError={onImageError}
                />
                {predicting && (
                  <>
                    <div className="scan-veil" aria-hidden="true" />
                    <span className="scan-ring" aria-hidden="true" />
                  </>
                )}
                <canvas
                  ref={camCanvasRef}
                  className={`preview-cam ${showCam ? 'is-on' : ''}`}
                  aria-hidden="true"
                  style={camRect ? {
                    left: `${camRect.left}px`,
                    top: `${camRect.top}px`,
                    width: `${camRect.width}px`,
                    height: `${camRect.height}px`,
                  } : undefined}
                />
              </div>
              <motion.button
                type="button"
                className="preview-clear"
                onClick={(e) => { e.stopPropagation(); onClear(); }}
                aria-label="Quitar imagen"
                title="Quitar imagen"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </motion.button>
            </motion.div>
          ) : (
            <motion.div
              key="hint"
              className="dropzone-hint"
              id="dropzone-hint"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="dropzone-field">
                <svg className="dropzone-icon" width="26" height="26" viewBox="0 0 32 32" fill="none" aria-hidden="true">
                  <path d="M16 4v16m0 0l-6-6m6 6l6-6M4 24h24" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {imageError ? (
                  <>
                    <p className="dropzone-title">No se pudo abrir la imagen</p>
                    <p className="dropzone-sub">Prueba con otro archivo</p>
                  </>
                ) : (
                  <>
                    <p className="dropzone-title">Arrastra una imagen dermatoscópica</p>
                    <p className="dropzone-sub">o haz clic para seleccionar un archivo</p>
                    <p className="dropzone-formats">JPEG · PNG · WebP · Máx.{"\u00A0"}10{"\u00A0"}MB</p>
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
