export const GITHUB_USER = import.meta.env.VITE_GITHUB_USER ?? 'abarriuso';
export const REPO_NAME = import.meta.env.VITE_REPO_NAME ?? 'melanoma-detection-vgg16';

export const DATASET_NAME = 'Melanoma Skin Cancer Dataset (10 000 imágenes)';
export const DATASET_URL =
  'https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images';

export const UMBRAL = 0.5;

export const MODELS = [
  {
    id: 'vgg16',
    name: 'VGG16',
    label: 'VGG16 (recomendado)',
    path: 'model/vgg16/model.json',
    temperature: 0.902,
    targetLayer: 'block5_conv3',
    version: '1.0.0',
    sizeMB: 15,
    auc: 0.9606,
    accuracy: 0.888,
    sens: 0.878,
    spec: 0.898,
  },
  {
    id: 'resnet50v2',
    name: 'ResNet50V2',
    label: 'ResNet50V2',
    path: 'model/resnet50v2/model.json',
    temperature: null,
    targetLayer: 'conv5_block3_conv',
    version: '1.0.0',
    sizeMB: 30,
    auc: null,
    accuracy: null,
    sens: null,
    spec: null,
  },
  {
    id: 'efficientnetv2s',
    name: 'EfficientNetV2S',
    label: 'EfficientNetV2S',
    path: 'model/efficientnetv2s/model.json',
    temperature: null,
    targetLayer: 'top_conv',
    version: '1.0.0',
    sizeMB: 15,
    auc: null,
    accuracy: null,
    sens: null,
    spec: null,
  },
];

export function getModel(id) {
  return MODELS.find((m) => m.id === id) || MODELS[0];
}
