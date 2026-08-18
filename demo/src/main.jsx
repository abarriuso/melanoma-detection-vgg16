import React from 'react';
import ReactDOM from 'react-dom/client';
import { MotionConfig } from 'framer-motion';
// Auto-host de fuentes (self-hosted via @fontsource). Sustituye al <link> a
// fonts.googleapis.com: misma tipografía sin round-trip a Google ni cookies.
// Sistema "instrumento de laboratorio": IBM Plex Sans (texto y titulares,
// 400/500/600/700 + itálica 400) e IBM Plex Mono (etiquetas y datos,
// 400/500/600). Solo importamos los pesos que el CSS usa y SOLO subset
// latin para reducir peso.
import '@fontsource/ibm-plex-sans/latin-400.css';
import '@fontsource/ibm-plex-sans/latin-400-italic.css';
import '@fontsource/ibm-plex-sans/latin-500.css';
import '@fontsource/ibm-plex-sans/latin-600.css';
import '@fontsource/ibm-plex-sans/latin-700.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@fontsource/ibm-plex-mono/latin-600.css';
import App from './App.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* Respeta prefers-reduced-motion en todas las animaciones de framer-motion */}
    <MotionConfig reducedMotion="user">
      <App />
    </MotionConfig>
  </React.StrictMode>
);
