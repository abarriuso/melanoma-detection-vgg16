// Genera demo/public/og.png a partir de un SVG.
// Script de uso ocasional: regenerar si cambia el branding, el copy o las
// métricas. `sharp` es una dependencia transitoria (no se versiona):
//
//   pnpm add -D sharp    # en la RAÍZ del repo
//   node scripts/gen-og.mjs
//
// Output: demo/public/og.png (1200×630, ~32 KB).
// Identidad "Atlas Dermatoscópico": papel hueso, tinta, acento verdigris,
// serif editorial y filetes finos — coherente con demo/src/index.css.
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
const IVORY = '#efe9dd';
const INK = '#1c1a17';
const ACCENT = '#0f5e5a';
const MUTED = '#6b655b';
const HAIRLINE = '#d8d2c4';

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="${W}" height="${H}" fill="${PAPER}"/>

  <!-- Filete superior + eyebrow -->
  <line x1="80" y1="96" x2="1120" y2="96" stroke="${HAIRLINE}" stroke-width="1"/>
  <text x="80" y="76" fill="${ACCENT}" font-family="Consolas, ui-monospace, monospace"
        font-size="20" font-weight="600" letter-spacing="6">
    ATLAS DERMATOSCÓPICO
  </text>
  <text x="1120" y="76" fill="${MUTED}" font-family="Consolas, ui-monospace, monospace"
        font-size="18" letter-spacing="3" text-anchor="end">
    INFERENCIA 100 % LOCAL
  </text>

  <!-- Título principal (serif editorial) -->
  <text x="80" y="245" fill="${INK}" font-family="Georgia, 'Times New Roman', serif"
        font-size="96" font-weight="400" letter-spacing="-2">
    Detección de <tspan font-style="italic" fill="${ACCENT}">melanoma</tspan>
  </text>

  <!-- Subtítulo -->
  <text x="80" y="315" fill="${MUTED}" font-family="Georgia, 'Times New Roman', serif"
        font-size="27" font-weight="400">
    EfficientNetV2S con fine-tuning. Clasificación benigno/maligno
  </text>
  <text x="80" y="355" fill="${MUTED}" font-family="Georgia, 'Times New Roman', serif"
        font-size="27" font-weight="400">
    ejecutada por completo en tu navegador.
  </text>

  <!-- Métricas: columnas separadas por filetes -->
  <g font-family="Consolas, ui-monospace, monospace">
    <line x1="80" y1="425" x2="1120" y2="425" stroke="${HAIRLINE}" stroke-width="1"/>

    <text x="80" y="465" fill="${MUTED}" font-size="15" letter-spacing="2">AUC EN TEST</text>
    <text x="80" y="510" fill="${INK}" font-size="40" font-weight="700">0.9742</text>

    <line x1="400" y1="440" x2="400" y2="525" stroke="${HAIRLINE}" stroke-width="1"/>
    <text x="440" y="465" fill="${MUTED}" font-size="15" letter-spacing="2">ACCURACY</text>
    <text x="440" y="510" fill="${INK}" font-size="40" font-weight="700">91.6 %</text>

    <line x1="700" y1="440" x2="700" y2="525" stroke="${HAIRLINE}" stroke-width="1"/>
    <text x="740" y="465" fill="${MUTED}" font-size="15" letter-spacing="2">SENSIBILIDAD</text>
    <text x="740" y="510" fill="${INK}" font-size="40" font-weight="700">88.2 %</text>

    <line x1="80" y1="545" x2="1120" y2="545" stroke="${HAIRLINE}" stroke-width="1"/>
  </g>

  <!-- Firma -->
  <text x="80" y="590" fill="${INK}" font-family="Georgia, 'Times New Roman', serif"
        font-size="21" font-weight="400">
    Adrián Barriuso Pizarro
  </text>
  <text x="1120" y="590" fill="${ACCENT}" font-family="Consolas, ui-monospace, monospace"
        font-size="17" font-weight="500" text-anchor="end">
    abarriuso.github.io/melanoma-detection-vgg16
  </text>
</svg>`;

await sharp(Buffer.from(svg))
  .png({ compressionLevel: 9, quality: 90 })
  .toFile(outFile);

console.log(`Generado: ${outFile}`);
