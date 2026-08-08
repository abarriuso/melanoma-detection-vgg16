export const GITHUB_USER = import.meta.env.VITE_GITHUB_USER ?? 'abarriuso';
export const REPO_NAME = import.meta.env.VITE_REPO_NAME ?? 'melanoma-detection-vgg16';

export const DATASET_NAME = 'Melanoma Skin Cancer Dataset (10 000 imágenes)';
export const DATASET_URL =
  'https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images';

// Segundo dataset (CC0), solo para variar las miniaturas de ejemplo del
// panel 1 — el modelo no se entrena con él. Ver scripts/download_examples_dataset.ps1.
export const EXAMPLES_DATASET2_NAME = 'Melanoma Detection Dataset (ISIC 2017)';
export const EXAMPLES_DATASET2_URL =
  'https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection';

export const UMBRAL = 0.5;

// Métricas medidas sobre el conjunto de test (1 000 imágenes, 500 por clase)
// en el entrenamiento conjunto de Kaggle (notebooks/entrenamiento_conjunto_kaggle.ipynb).
// EfficientNetV2S es el mejor en AUC, accuracy, sensibilidad y especificidad.
export const MODELS = [
  {
    id: 'efficientnetv2s',
    name: 'EfficientNetV2S',
    label: 'EfficientNetV2S (recomendado)',
    path: 'model/efficientnetv2s/model.json',
    temperature: 1.1836,
    targetLayer: 'top_conv',
    version: '2.0.0',
    sizeMB: 21,
    auc: 0.9742,
    accuracy: 0.916,
    sens: 0.882,
    spec: 0.95,
  },
  {
    id: 'resnet50v2',
    name: 'ResNet50V2',
    label: 'ResNet50V2',
    path: 'model/resnet50v2/model.json',
    temperature: 1.0221,
    targetLayer: 'post_relu',
    version: '2.0.0',
    sizeMB: 25,
    auc: 0.9727,
    accuracy: 0.909,
    sens: 0.878,
    spec: 0.94,
  },
  {
    id: 'vgg16',
    name: 'VGG16',
    label: 'VGG16',
    path: 'model/vgg16/model.json',
    temperature: 1.3359,
    targetLayer: 'block5_conv3',
    version: '2.0.0',
    sizeMB: 15,
    auc: 0.9712,
    accuracy: 0.903,
    sens: 0.862,
    spec: 0.944,
  },
];

export function getModel(id) {
  return MODELS.find((m) => m.id === id) || MODELS[0];
}
