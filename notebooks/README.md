# notebooks/

Cuadernos de entrenamiento independientes para cada arquitectura.

| Notebook | Modelo | Estado |
|----------|--------|--------|
| `vgg16.ipynb` | VGG16 | Entrenado, AUC 0.9606 |
| `resnet50v2.ipynb` | ResNet50V2 | Pendiente de entrenar |
| `efficientnetv2s.ipynb` | EfficientNetV2S | Pendiente de entrenar |

## Uso

1. Sube cada notebook a Google Colab (GPU T4+)
2. Ejecuta todas las celdas (~30-45 min por modelo)
3. El modelo TF.js cuantizado se guarda en `models/<id>/tfjs/`
4. Copia a `demo/public/model/<id>/`:

```bash
cp -r models/vgg16/tfjs/*        demo/public/model/vgg16/
cp -r models/resnet50v2/tfjs/*   demo/public/model/resnet50v2/
cp -r models/efficientnetv2s/tfjs/* demo/public/model/efficientnetv2s/
```

Cada notebook es autónomo: tiene su propio pipeline de datos, entrenamiento,
evaluación, calibración (temperature scaling) y conversión a TF.js.
