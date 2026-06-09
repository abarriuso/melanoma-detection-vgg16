// Genera los iconos PWA (192/512 y maskable 512) a partir de
// demo/public/favicon.svg. Sharp es transitoria (no se versiona):
//
//   npm install --no-save sharp    # en la RAÍZ del repo
//   node scripts/gen-icons.mjs
import sharp from 'sharp';
import { readFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const publicDir = resolve(here, '..', 'demo', 'public');
mkdirSync(publicDir, { recursive: true });

const svg = readFileSync(resolve(publicDir, 'favicon.svg'));

for (const size of [192, 512]) {
  const out = resolve(publicDir, `icon-${size}.png`);
  await sharp(svg, { density: 384 })
    .resize(size, size)
    .png({ compressionLevel: 9 })
    .toFile(out);
  console.log(`Generado: ${out}`);
}

// Icono maskable: usa el mismo arte pero con padding interior del ~15%
// para que iconos Android con recorte circular no corten el contenido.
{
  const size = 512;
  const padded = Math.round(size * 0.7);
  const out = resolve(publicDir, `icon-maskable-${size}.png`);
  const inner = await sharp(svg).resize(padded, padded).png().toBuffer();
  await sharp({
    create: { width: size, height: size, channels: 4, background: '#fef3c7' },
  })
    .composite([{ input: inner, gravity: 'center' }])
    .png({ compressionLevel: 9 })
    .toFile(out);
  console.log(`Generado: ${out}`);
}
