import React from 'react';
import ReactDOM from 'react-dom/client';
// Auto-host de fuentes (self-hosted via @fontsource). Sustituye al <link> a
// fonts.googleapis.com: misma tipografía sin round-trip a Google ni cookies.
// Solo importamos los pesos que el CSS usa (400/500/600/700 en Inter, 400/500
// en JetBrains Mono) y SOLO subset latin para reducir ~200 KB.
import '@fontsource/inter/latin-400.css';
import '@fontsource/inter/latin-500.css';
import '@fontsource/inter/latin-600.css';
import '@fontsource/inter/latin-700.css';
import '@fontsource/jetbrains-mono/latin-400.css';
import '@fontsource/jetbrains-mono/latin-500.css';
import App from './App.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
