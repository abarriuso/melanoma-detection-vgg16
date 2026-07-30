// Repuebla demo/public/samples/{benign,malignant}/ con N imágenes random
// del conjunto de test. Regenera manifest.json. Script idempotente:
// si se ejecuta dos veces seguidas con el mismo N, normalmente el set
// resultante será distinto (selección random nueva). El semillado fijo
// se omite a propósito para que cada regeneración refresque el pool.
//
//   node scripts/refresh-samples.mjs              # 60 + 60 por defecto
//   node scripts/refresh-samples.mjs --count 80   # 80 + 80
//
// Requiere el conjunto de test del dataset descargado. Se acepta tanto
// dataset/ (donde lo deja scripts/download_dataset.ps1) como archive/
// (nombre de la carpeta si descomprimes el zip de Kaggle a mano).
import { readdirSync, copyFileSync, rmSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const dstRoot = join(root, 'demo', 'public', 'samples');

const CANDIDATE_ROOTS = [
  join(root, 'dataset', 'melanoma_cancer_dataset', 'test'),
  join(root, 'archive', 'melanoma_cancer_dataset', 'test'),
];
const srcRoot = CANDIDATE_ROOTS.find((p) => existsSync(p));

const arg = process.argv.indexOf('--count');
const COUNT_PER_CLASS = arg !== -1 ? Number(process.argv[arg + 1]) : 60;

if (!srcRoot) {
  console.error(
    'No se encuentra el conjunto de test. Se ha buscado en:\n' +
      CANDIDATE_ROOTS.map((p) => `  - ${p}`).join('\n') +
      '\nDescarga el dataset primero: pwsh ./scripts/download_dataset.ps1',
  );
  process.exit(1);
}

function pickRandom(arr, n) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, Math.min(n, a.length));
}

function refresh(klass) {
  const src = join(srcRoot, klass);
  const dst = join(dstRoot, klass);
  rmSync(dst, { recursive: true, force: true });
  mkdirSync(dst, { recursive: true });
  const all = readdirSync(src).filter((f) => /\.(jpe?g|png|webp)$/i.test(f));
  const picked = pickRandom(all, COUNT_PER_CLASS);
  picked.forEach((f) => copyFileSync(join(src, f), join(dst, f)));
  console.log(`  ${klass}: ${picked.length}/${all.length}`);
  return picked.sort();
}

console.log(`Repoblando ${dstRoot}`);
const malignant = refresh('malignant');
const benign = refresh('benign');

const manifest = { malignant, benign };
writeFileSync(join(dstRoot, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`Manifest actualizado: ${malignant.length + benign.length} muestras.`);
