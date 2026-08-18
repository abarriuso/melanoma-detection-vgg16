// Genera demo/public/og.png a partir de un SVG.
// Script de uso ocasional: regenerar si cambia el branding, el copy o las
// métricas. `sharp` está declarada como devDependency en el package.json
// de la RAÍZ del repo:
//
//   npm install            # en la RAÍZ del repo (instala sharp)
//   node scripts/gen-og.mjs
//
// Output: demo/public/og.png (1200×630, ~32 KB).
// Identidad: dirección "instrumento de laboratorio":
// papel hueso, tinta, acento verdigris, sans semibold para el titular y mono
// con tracking moderado para etiquetas y cifras — coherente con
// demo/src/index.css. sharp/librsvg usa fuentes del sistema: IBM Plex no está
// instalada en Windows, así que se usa un stack sans genérico (Segoe UI) y
// mono de sistema (Consolas) como aproximación razonable.
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

// Paleta (tokens de demo/src/index.css)
const PAPER = '#f7f4ee';
const INK = '#1c1a17';
const ACCENT = '#0f5e5a';
const MUTED = '#6b655b';
const HAIRLINE = '#d8d2c4';

// Stacks de sistema (librsvg no conoce IBM Plex en esta máquina)
const SANS = `'Segoe UI', Arial, Helvetica, sans-serif`;
const MONO = `Consolas, ui-monospace, monospace`;

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="${W}" height="${H}" fill="${PAPER}"/>

  <!-- Filete superior + wordmark -->
  <line x1="80" y1="96" x2="1120" y2="96" stroke="${HAIRLINE}" stroke-width="1"/>
  <text x="80" y="76" fill="${ACCENT}" font-family="${MONO}"
        font-size="20" font-weight="600" letter-spacing="2">
    MELANOMA-DETECTION
  </text>
  <text x="1120" y="76" fill="${MUTED}" font-family="${MONO}"
        font-size="18" letter-spacing="1.5" text-anchor="end">
    INFERENCIA 100 % LOCAL
  </text>

  <!-- Título principal (sans semibold, "melanoma" en acento roman) -->
  <text x="80" y="245" fill="${INK}" font-family="${SANS}"
        font-size="88" font-weight="600" letter-spacing="-2">
    Detección de <tspan fill="${ACCENT}">melanoma</tspan>
  </text>

  <!-- Subtítulo -->
  <text x="80" y="315" fill="${MUTED}" font-family="${SANS}"
        font-size="27" font-weight="400">
    EfficientNetV2S con fine-tuning. Clasificación benigno/maligno
  </text>
  <text x="80" y="355" fill="${MUTED}" font-family="${SANS}"
        font-size="27" font-weight="400">
    ejecutada por completo en tu navegador.
  </text>

  <!-- Métricas: columnas separadas por filetes -->
  <g font-family="${MONO}">
    <line x1="80" y1="425" x2="1120" y2="425" stroke="${HAIRLINE}" stroke-width="1"/>

    <text x="80" y="465" fill="${MUTED}" font-size="15" letter-spacing="1.5">AUC EN TEST</text>
    <text x="80" y="510" fill="${INK}" font-size="40" font-weight="700">0.971</text>

    <line x1="400" y1="440" x2="400" y2="525" stroke="${HAIRLINE}" stroke-width="1"/>
    <text x="440" y="465" fill="${MUTED}" font-size="15" letter-spacing="1.5">ACCURACY</text>
    <text x="440" y="510" fill="${INK}" font-size="40" font-weight="700">87.1 %</text>

    <line x1="700" y1="440" x2="700" y2="525" stroke="${HAIRLINE}" stroke-width="1"/>
    <text x="740" y="465" fill="${MUTED}" font-size="15" letter-spacing="1.5">SENSIBILIDAD</text>
    <text x="740" y="510" fill="${INK}" font-size="40" font-weight="700">90.0 %</text>

    <line x1="80" y1="545" x2="1120" y2="545" stroke="${HAIRLINE}" stroke-width="1"/>
  </g>

  <!-- Firma -->
  <text x="80" y="590" fill="${INK}" font-family="${SANS}"
        font-size="21" font-weight="500">
    Adrián Barriuso Pizarro
  </text>
  <text x="1120" y="590" fill="${ACCENT}" font-family="${MONO}"
        font-size="17" font-weight="500" text-anchor="end">
    abarriuso.github.io/melanoma-detection-vgg16
  </text>
</svg>`;

await sharp(Buffer.from(svg))
  .png({ compressionLevel: 9, quality: 90 })
  .toFile(outFile);

console.log(`Generado: ${outFile}`);
