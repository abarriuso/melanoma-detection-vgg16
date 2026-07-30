// Repuebla demo/public/samples/{benign,malignant}/ con N imágenes random
// del conjunto de test. Regenera manifest.json. Script idempotente:
// si se ejecuta dos veces seguidas con el mismo N, normalmente el set
// resultante será distinto (selección random nueva). El semillado fijo
// se omite a propósito para que cada regeneración refresque el pool.
//
//   node scripts/refresh-samples.mjs              # 60 + 60 por defecto (por dataset)
//   node scripts/refresh-samples.mjs --count 80   # 80 + 80
//
// Requiere el conjunto de test del dataset principal descargado. Se acepta
// tanto dataset/ (scripts/download_dataset.ps1) como archive/ (nombre de la
// carpeta si descomprimes el zip de Kaggle a mano).
//
// Si además existe dataset2/ (scripts/download_examples_dataset.ps1, el
// "Melanoma Detection Dataset" de wanderdust / ISIC 2017), se suman también
// sus imágenes al pool de ejemplos: es un dataset de 3 clases, así que
// melanoma -> maligno y {nevus, seborrheic_keratosis} -> benigno (ambas son
// lesiones benignas). Los ficheros se prefijan con "isic17_" para no
// chocar con los nombres del dataset principal. Si dataset2/ no está, se
// ignora sin más: el pool sale solo del dataset principal, como siempre.
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

const EXAMPLES_CANDIDATE_ROOTS = [
  join(root, 'dataset2', 'skin-lesions'),
  join(root, 'archive2', 'skin-lesions'),
];
const examplesRoot = EXAMPLES_CANDIDATE_ROOTS.find((p) => existsSync(p));

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

// Recoge ficheros de varias subcarpetas de origen (las 3 clases de ISIC17
// mapeadas a benigno/maligno) y los copia con prefijo a demo/public/samples/<klass>/,
// sumándolos al array `into` (que ya trae lo del dataset principal).
function addFromExamplesDataset(klass, subfolders, into) {
  if (!examplesRoot) return into;
  const dst = join(dstRoot, klass);
  mkdirSync(dst, { recursive: true });
  const perSubfolder = Math.ceil(COUNT_PER_CLASS / subfolders.length);
  const added = [];
  for (const split of ['test', 'valid', 'train']) {
    for (const sub of subfolders) {
      const dir = join(examplesRoot, split, sub);
      if (!existsSync(dir)) continue;
      const all = readdirSync(dir).filter((f) => /\.(jpe?g|png|webp)$/i.test(f));
      const picked = pickRandom(all, perSubfolder);
      for (const f of picked) {
        const name = `isic17_${sub}_${f}`;
        copyFileSync(join(dir, f), join(dst, name));
        added.push(name);
      }
    }
    if (added.length >= COUNT_PER_CLASS) break; // ya hay bastante con test/valid
  }
  console.log(`  ${klass} (ISIC17): +${added.length}`);
  return [...into, ...added];
}

console.log(`Repoblando ${dstRoot}`);
let malignant = refresh('malignant');
let benign = refresh('benign');

if (examplesRoot) {
  console.log(`Sumando "Melanoma Detection Dataset" (ISIC17) desde ${examplesRoot}`);
  malignant = addFromExamplesDataset('malignant', ['melanoma'], malignant);
  benign = addFromExamplesDataset('benign', ['nevus', 'seborrheic_keratosis'], benign);
} else {
  console.log('(dataset2/ no encontrado; solo se usa el dataset principal. Ver scripts/download_examples_dataset.ps1 para sumar el segundo.)');
}

malignant = malignant.sort();
benign = benign.sort();

const manifest = { malignant, benign };
writeFileSync(join(dstRoot, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`Manifest actualizado: ${malignant.length + benign.length} muestras.`);
