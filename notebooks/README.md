# notebooks/

Cuadernos de entrenamiento autónomos, uno por arquitectura. Pensados para
**Google Colab con GPU T4**.

| Notebook | Backbone | Grad-CAM target | Estado |
|----------|----------|-----------------|--------|
| `vgg16.ipynb` | VGG16 | `block5_conv3` | Entrenado (AUC 0.96) |
| `resnet50v2.ipynb` | ResNet50V2 | `post_relu` | Pendiente de entrenar |
| `efficientnetv2s.ipynb` | EfficientNetV2S | `top_conv` | Pendiente de entrenar |

Cada notebook es independiente: pipeline de datos, arquitectura, entrenamiento
en dos fases, evaluación, métricas clínicas (sensibilidad/especificidad,
análisis de umbral), calibración (temperature scaling), MC Dropout, Grad-CAM y
Grad-CAM++, curva Precision-Recall, TTA, análisis de falsos negativos y
exportación a TF.js.

## Diseño

- El modelo Keras recibe la imagen en **[0, 1] RGB** (lo mismo que la demo
  TF.js: `resize → /255`) y **hornea el preprocesado del backbone dentro del
  grafo**, así el modelo exportado es autocontenido.
- El backbone se construye con `input_tensor=` para obtener un grafo plano y
  poder aplicar Grad-CAM sobre la capa convolucional objetivo.
- La exportación a TF.js usa **Keras 2 legacy** (`tf-keras` +
  `TF_USE_LEGACY_KERAS=1`, fijado en la primera celda) porque `tensorflowjs`
  aún no soporta Keras 3 de forma fiable.

## Uso

1. Sube el dataset a tu Drive:
   `MyDrive/melanoma_cancer_dataset/{train,test}/{benign,malignant}/*.jpg`.
2. Abre el notebook en Colab y activa la GPU:
   **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**.
3. Ejecuta todas las celdas (~30-45 min por modelo). Si Colab pide reiniciar el
   entorno tras la instalación de la primera celda, hazlo y reejecuta.
4. El modelo se guarda en Drive: `MyDrive/melanoma_model/<id>/`
   (`.keras` + carpeta `tfjs/` cuantizada a uint8).

Para desplegar en la demo, descarga la carpeta `tfjs/` y copia su contenido:

```bash
cp -r <descarga>/vgg16/tfjs/*          demo/public/model/vgg16/
cp -r <descarga>/resnet50v2/tfjs/*     demo/public/model/resnet50v2/
cp -r <descarga>/efficientnetv2s/tfjs/* demo/public/model/efficientnetv2s/
```

Luego actualiza `temperature` (la imprime la celda de calibración) y, tras
entrenar, las métricas del modelo en `demo/src/lib/constants.js`.
