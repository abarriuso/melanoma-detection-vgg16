# models/

Modelos entrenados (.keras) y convertidos a TF.js (cuantizados uint8).

| Modelo | Carpeta | Estado |
|--------|---------|--------|
| EfficientNetV2S | `efficientnetv2s/` | AUC 0.9742 · acc 91.6 % · sens 88.2 % · spec 95.0 % · T 1.184 — **servido por defecto** |
| ResNet50V2 | `resnet50v2/` | AUC 0.9727 · acc 90.9 % · sens 87.8 % · spec 94.0 % · T 1.022 |
| VGG16 | `vgg16/` | AUC 0.9712 · acc 90.3 % · sens 86.2 % · spec 94.4 % · T 1.336 · ~15 MB |

Métricas del entrenamiento conjunto en Kaggle
(`notebooks/entrenamiento_conjunto_kaggle.ipynb`, test de 1 000 imágenes).

Cada modelo se entrena desde su notebook en `notebooks/` y guarda tanto el
`.keras` original como la versión TF.js (en `tfjs/` dentro de cada carpeta).
Si el entorno de entrenamiento no tenía `tensorflowjs` (Kaggle), convierte los
`.keras` con `scripts/convert_kaggle_output.py` antes de desplegar la demo.
