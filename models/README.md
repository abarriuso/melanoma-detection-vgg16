# models/

Modelos entrenados (.keras) y convertidos a TF.js (cuantizados uint8).

| Modelo | Carpeta | Estado |
|--------|---------|--------|
| VGG16 | `vgg16/` | AUC 0.9606, ~15 MB |
| ResNet50V2 | `resnet50v2/` | Pendiente |
| EfficientNetV2S | `efficientnetv2s/` | Pendiente |

Cada modelo se entrena desde su notebook en `notebooks/` y guarda tanto el
`.keras` original como la versión TF.js (en `tfjs/` dentro de cada carpeta).
