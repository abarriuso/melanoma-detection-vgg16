// Genera demo/public/og.png a partir de un SVG.
// Script de uso ocasional: regenerar si cambia el branding, el copy o las
// métricas. `sharp` es una dependencia transitoria (no se versiona):
//
//   npm install --no-save sharp    # en la RAÍZ del repo
//   node scripts/gen-og.mjs
//
// Output: demo/public/og.png (1200×630, ~32 KB).
import sharp from 'sharp';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, '..', 'demo', 'public');
const outFile = resolve(outDir, 'og.png');
mkdirSync(outDir, { recursive: true });

const W = 1200;
const H = 630;

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0c1018"/>
      <stop offset="100%" stop-color="#131a26"/>
    </linearGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#5eead4"/>
      <stop offset="100%" stop-color="#2dd4bf"/>
    </linearGradient>
  </defs>

  <rect width="${W}" height="${H}" fill="url(#bg)"/>

  <!-- Línea acento superior -->
  <rect x="80" y="80" width="120" height="3" fill="url(#accent)"/>

  <!-- Eyebrow / kicker -->
  <text x="80" y="135" fill="#76859c" font-family="Inter, system-ui, sans-serif"
        font-size="22" font-weight="500" letter-spacing="3">
    DEEP LEARNING · DERMATOLOGÍA
  </text>

  <!-- Título principal -->
  <text x="80" y="225" fill="#e7ecf3" font-family="Inter, system-ui, sans-serif"
        font-size="68" font-weight="700" letter-spacing="-2">
    Clasificación de
  </text>
  <text x="80" y="305" fill="#e7ecf3" font-family="Inter, system-ui, sans-serif"
        font-size="68" font-weight="700" letter-spacing="-2">
    lesiones cutáneas
  </text>

  <!-- Subtítulo -->
  <text x="80" y="375" fill="#8593a6" font-family="Inter, system-ui, sans-serif"
        font-size="28" font-weight="400">
    VGG16 + fine-tuning. Inferencia 100% en el navegador.
  </text>

  <!-- Stack de badges -->
  <g font-family="JetBrains Mono, ui-monospace, monospace" font-weight="500">
    <!-- AUC -->
    <rect x="80" y="445" width="200" height="64" rx="2" fill="rgba(94,234,212,0.12)" stroke="#5eead4" stroke-width="1"/>
    <text x="100" y="478" fill="#76859c" font-size="14" letter-spacing="1">AUC</text>
    <text x="100" y="498" fill="#5eead4" font-size="26" font-weight="600">0.961</text>

    <!-- Accuracy -->
    <rect x="300" y="445" width="200" height="64" rx="2" fill="rgba(94,234,212,0.06)" stroke="#1f2937" stroke-width="1"/>
    <text x="320" y="478" fill="#76859c" font-size="14" letter-spacing="1">ACCURACY</text>
    <text x="320" y="498" fill="#e7ecf3" font-size="26" font-weight="600">88.8 %</text>

    <!-- Sensibilidad -->
    <rect x="520" y="445" width="240" height="64" rx="2" fill="rgba(94,234,212,0.06)" stroke="#1f2937" stroke-width="1"/>
    <text x="540" y="478" fill="#76859c" font-size="14" letter-spacing="1">SENSIBILIDAD</text>
    <text x="540" y="498" fill="#e7ecf3" font-size="26" font-weight="600">87.8 %</text>
  </g>

  <!-- Footer / firma -->
  <line x1="80" y1="555" x2="1120" y2="555" stroke="#1f2937" stroke-width="1"/>
  <text x="80" y="590" fill="#76859c" font-family="Inter, system-ui, sans-serif"
        font-size="20" font-weight="500">
    Adrián Barriuso Pizarro
  </text>
  <text x="1120" y="590" fill="#5eead4" font-family="JetBrains Mono, ui-monospace, monospace"
        font-size="18" font-weight="500" text-anchor="end">
    abarriuso.github.io/melanoma-detection-vgg16
  </text>
</svg>`;

await sharp(Buffer.from(svg))
  .png({ compressionLevel: 9, quality: 90 })
  .toFile(outFile);

console.log(`Generado: ${outFile}`);
