// Constantes del proyecto. Configurables via Vite env variables (VITE_GITHUB_USER, VITE_REPO_NAME).
// Ver vite.config.js para valores por defecto.
export const GITHUB_USER = import.meta.env.VITE_GITHUB_USER ?? 'abarriuso';
export const REPO_NAME = import.meta.env.VITE_REPO_NAME ?? 'melanoma-detection-vgg16';

// Dataset de origen (imágenes de ejemplo y de la clasificación en lote).
export const DATASET_NAME = 'Melanoma Skin Cancer Dataset (10 000 imágenes)';
export const DATASET_URL =
  'https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images';

// Umbral de decisión: score >= 0.5 → maligno. Coincide con el del notebook.
export const UMBRAL = 0.5;
