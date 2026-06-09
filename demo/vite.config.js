import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Variables de entorno expuestas al cliente (prefijo VITE_)
// Permiten configurar usuario/repo de GitHub sin tocar el código.
// eslint-disable-next-line no-undef
const GITHUB_USER = process.env.VITE_GITHUB_USER ?? 'abarriuso';
// eslint-disable-next-line no-undef
const REPO_NAME = process.env.VITE_REPO_NAME ?? 'melanoma-detection-vgg16';

// CSP: en dev necesitamos 'unsafe-eval' para HMR; en prod no.
function cspPlugin() {
  return {
    name: 'csp-plugin',
    transformIndexHtml(html, { server }) {
      // eslint-disable-next-line no-undef
      const isDev = server ? true : process.env.NODE_ENV !== 'production';
      const scriptSrc = isDev ? "script-src 'self' 'unsafe-eval';" : "script-src 'self';";
      const csp = `default-src 'self'; ${scriptSrc} style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; worker-src 'self' blob:; base-uri 'self'; form-action 'self'; frame-ancestors 'self'; object-src 'none'; child-src 'none'; media-src 'none'; upgrade-insecure-requests;`;
      return html.replace('</head>', `  <meta http-equiv="Content-Security-Policy" content="${csp}">\n</head>`);
    },
  };
}

export default defineConfig({
  define: {
    'import.meta.env.VITE_GITHUB_USER': JSON.stringify(GITHUB_USER),
    'import.meta.env.VITE_REPO_NAME': JSON.stringify(REPO_NAME),
  },
  plugins: [
    cspPlugin(),
    react(),
    // PWA: registra un service worker que cachea index + assets + modelo
    // para reuso offline. Estrategia "registerType: autoUpdate" descarga
    // la nueva versión en background y activa al recargar.
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'og.png'],
      manifest: {
        name: 'Detección de melanoma — VGG16 fine-tuning',
        short_name: 'Melanoma VGG16',
        description:
          'Clasificación dermoscópica de lesiones cutáneas con VGG16 fine-tuned. Inferencia 100% client-side con TensorFlow.js.',
        lang: 'es',
        start_url: `/${REPO_NAME}/`,
        scope: `/${REPO_NAME}/`,
        display: 'standalone',
        background_color: '#0d1117',
        theme_color: '#0d1117',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // Por defecto Workbox descarta archivos >2 MB. El modelo TF.js
        // (.json + 4 shards .bin de ~4 MB cada uno) y el chunk de tfjs
        // (~1.5 MB) deben caber, así que subimos el techo a 6 MB.
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
        globPatterns: ['**/*.{js,css,html,svg,png,ico,webmanifest,json,bin}'],
        // Las muestras del test set no necesitan estar precacheadas (se
        // descargan al usarse). Reducir el tamaño del install.
        globIgnores: ['**/samples/benign/**', '**/samples/malignant/**'],
      },
    }),
  ],
  // `base` prefija todas las rutas de assets con el nombre del repo.
  // Necesario porque GitHub Pages sirve en un subdirectorio:
  //   https://abarriuso.github.io/melanoma-detection-vgg16/
  // Sin esto, las rutas absolutas (/assets/...) apuntarían a la raíz
  // del dominio (github.io/) y no encontrarían los archivos.
  base: `/${REPO_NAME}/`,
  build: {
    // TF.js es la mayoría del bundle (~1.4 MB). Aislándolo en su propio
    // chunk, React puede pintar la UI antes de que TF.js termine de
    // descargarse/parsearse y el caché del navegador lo reutiliza entre
    // recargas aunque cambie el código de la app.
    rollupOptions: {
      output: {
        manualChunks: {
          tfjs: ['@tensorflow/tfjs'],
          react: ['react', 'react-dom'],
          resultsGallery: ['./src/ResultsGallery'],
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    css: true,
    include: ['src/**/*.{test,spec}.{js,jsx}'],
  },
});
