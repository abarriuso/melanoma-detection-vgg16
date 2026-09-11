# Detección de melanoma con transfer learning (VGG16 · ResNet50V2 · EfficientNetV2S)

Clasificación binaria de imágenes dermatoscópicas (benigno / maligno) mediante
*transfer learning*. Se comparan tres arquitecturas de referencia con un
protocolo idéntico y se sirve la mejor en una demo web que ejecuta la inferencia
en el navegador del usuario.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)
![React](https://img.shields.io/badge/React-18-61DAFB)
![TF.js](https://img.shields.io/badge/TF.js-Client--side-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

**Demo en vivo:** https://abarriuso.github.io/melanoma-detection-vgg16/

> **Aviso.** Este es un proyecto académico y una prueba de concepto. **No es un
> dispositivo médico, no está validado clínicamente y no debe usarse para tomar
> decisiones de salud.** Sus resultados se obtienen sobre un único conjunto de
> datos público y no representan el rendimiento en condiciones clínicas reales.
> Ante cualquier lesión sospechosa, consulta a un dermatólogo. Texto completo en
> [DISCLAIMER.md](DISCLAIMER.md).

---

## Índice

- [Motivación y alcance](#motivación-y-alcance)
- [Datos](#datos)
- [Metodología](#metodología)
- [Modelos comparados](#modelos-comparados)
- [Resultados](#resultados)
- [Interpretabilidad y calibración](#interpretabilidad-y-calibración)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación y uso](#instalación-y-uso)
- [Limitaciones](#limitaciones)
- [Consideraciones éticas y clínicas](#consideraciones-éticas-y-clínicas)
- [Referencias](#referencias)
- [Licencia y datos](#licencia-y-datos)
- [Autor](#autor)

---

## Motivación y alcance

El melanoma es uno de los cánceres de piel más agresivos, y la evidencia
clínica asocia su detección temprana con un mejor pronóstico [1]. En la última
década, las redes neuronales convolucionales (CNN) han demostrado capacidad para
clasificar lesiones cutáneas a partir de imágenes, en algunos estudios a un nivel
comparable al de dermatólogos sobre conjuntos de test acotados [2]. Conviene
subrayar que esos resultados se obtienen en condiciones controladas y no
equivalen a validación clínica.

El alcance de este trabajo es **educativo**: reproducir, con rigor y de forma
transparente, un flujo completo de clasificación binaria (benigno / maligno) —
desde el entrenamiento hasta el despliegue— y comparar tres arquitecturas
conocidas. No se persigue un sistema de diagnóstico ni se afirma utilidad
clínica.

## Datos

Se utiliza el *Melanoma Skin Cancer Dataset* (10 000 imágenes, licencia CC0)
publicado en Kaggle [3]. El conjunto está dividido en entrenamiento y test, con
una partición interna 80/20 para validación. El test contiene 1 000 imágenes,
500 por clase.

**Sesgos conocidos del conjunto** (relevantes para interpretar los resultados):

- Está **balanceado artificialmente al 50/50**. La prevalencia real del melanoma
  en la práctica clínica es muy inferior, por lo que el valor predictivo positivo
  observado aquí no es trasladable a un entorno real [4].
- No documenta la distribución de fototipos de piel, el equipamiento ni las
  condiciones de captura. Un modelo entrenado así puede generalizar mal a
  poblaciones o dispositivos distintos.
- Es un único origen de datos: sin validación externa no puede estimarse la
  generalización fuera de dominio.

## Metodología

El protocolo es común a las tres arquitecturas para que la comparación sea justa:

- **Transfer learning** desde pesos preentrenados en ImageNet [5], técnica
  estándar cuando el conjunto objetivo es pequeño [6].
- **Entrenamiento en dos fases:** (1) extracción de características con el
  *backbone* congelado (RMSprop, lr 1e-4); (2) *fine-tuning* de las capas altas
  (Adam, lr 1e-5–1e-6). *Callbacks*: `ModelCheckpoint` (mejor `val_loss`),
  `EarlyStopping` y `ReduceLROnPlateau`.
- **Data augmentation** solo en entrenamiento: volteos, rotación, zoom,
  traslación, brillo y contraste.
- **Ponderación de clases** `class_weight = {0: 1.0, 1: 1.3}` para penalizar más
  el error sobre la clase maligna y favorecer la sensibilidad [4].
- **Reproducibilidad:** semilla fija (42), entrada 224×224, mismo pipeline de
  datos para los tres modelos.

Cada modelo **hornea su preprocesado específico dentro del grafo**, de modo que
el modelo exportado es autocontenido y el cliente no necesita lógica por modelo.

El entrenamiento se realizó en Kaggle (GPU T4) mediante un notebook conjunto que
entrena las tres arquitecturas de una sola ejecución
([`notebooks/entrenamiento_conjunto_kaggle.ipynb`](notebooks/entrenamiento_conjunto_kaggle.ipynb)).

## Modelos comparados

| Modelo | Preprocesado (dentro del grafo) | Fine-tuning (fase 2) | LR fase 2 | Capa Grad-CAM | Referencia |
|--------|----------------------------------|----------------------|:---------:|---------------|-----------|
| **EfficientNetV2S** | `Rescaling(255)` + preprocesado interno de Keras | ~50 % de capas (BatchNorm congelado) | 1e-5 | `top_conv` | Tan & Le, 2021 [7] |
| ResNet50V2 | `Rescaling(2, offset=-1)` → [-1, 1] | ~80 capas no-BN (BatchNorm congelado) | 1e-6 | `post_relu` | He et al., 2016 [8] |
| VGG16 | `preprocess_input(x·255)` (BGR + medias Caffe) | Últimas 4 capas (bloque 5) | 1e-5 | `block5_conv3` | Simonyan & Zisserman, 2015 [9] |

En ResNet50V2 y EfficientNetV2S se congela `BatchNorm` durante el *fine-tuning*:
sus estadísticas se degradan con facilidad al reentrenar con lotes pequeños.

## Resultados

Métricas sobre el conjunto de test **limpio** (774 imágenes tras excluir 226
contaminadas por duplicados o cuasi-duplicados en train; ver
[`scripts/dedup_test.py`](scripts/dedup_test.py)). El modelo servido en la demo
(TF.js float32) se evalúa con el mismo preprocesado que el cliente.

| Modelo | Accuracy | AUC | Sensibilidad | Especificidad | VPP (maligno) | F1 macro | T | FN |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EfficientNetV2S** | **87.1 %** | **0.971** | **90.0 %** | **84.4 %** | **85.2 %** | **0.88** | 1.184 | **37** |
| ResNet50V2 | 89.1 % | 0.969 | 84.6 % | 93.3 % | 92.1 % | 0.89 | 1.022 | 57 |
| VGG16 | 88.5 % | 0.957 | 82.5 % | 94.0 % | 92.7 % | 0.88 | 1.336 | 65 |

*AUC para EfficientNetV2S y ResNet50V2 se estima a partir del modelo `.keras`
original; el AUC de VGG16 se calcula directamente sobre el test limpio con el
modelo TF.js. Sensibilidad = recall de la clase maligna. VPP = valor predictivo
positivo = precisión sobre malignos. T = temperatura de calibración. FN =
melanomas no detectados. Precisión, F1 y matrices de confusión se derivan del
umbral 0.5.*

> **Nota sobre la contaminación del test.** Una auditoría posterior detectó que
> ~23 % del test original (226/1 000 imágenes) tiene cuasi-duplicados en train
> (hash perceptual aHash 16×16, distancia de Hamming ≤ 12). Las métricas
> originales sobre las 1 000 imágenes eran una cota optimista: EfficientNetV2S
> 91.6 %, ResNet50V2 90.9 %, VGG16 90.3 %. Las cifras de la tabla superior son
> la evaluación honesta sobre el subconjunto limpio.

### Matriz de confusión — EfficientNetV2S (test limpio)

|  | Pred. Benigno | Pred. Maligno |
|---|:---:|:---:|
| **Real Benigno** | 340 (TN) | 63 (FP) |
| **Real Maligno** | 37 (FN) | 334 (TP) |

### Cómo leer estos números (con cautela)

- **Las diferencias entre los tres modelos son pequeñas** (AUC 0.957–0.971).
  Al tratarse de **una única ejecución de entrenamiento**, sin validación
  cruzada ni intervalos de confianza, esas diferencias podrían caer dentro de la
  variabilidad aleatoria entre ejecuciones. No debe concluirse que un modelo es
  categóricamente "mejor" a partir de estas cifras.
- El test original contenía ~23 % de imágenes con duplicados o
  cuasi-duplicados en train; al excluirlas, la sensibilidad y la especificidad
  varían de forma asimétrica porque las imágenes eliminadas no son una muestra
  aleatoria (tienen más probabilidad de ser benignas o malignas según la clase
  que predominaba entre los duplicados).
- Los **37 falsos negativos** (melanomas clasificados como benignos) son el error
  clínicamente más grave. La ponderación de clases empuja hacia la sensibilidad a
  costa de más falsos positivos; en un escenario real convendría además bajar el
  umbral de decisión por debajo de 0.5.
- El VPP (~85 %) está inflado por el balance 50/50 del test. Con la prevalencia
  real (mucho menor), el VPP sería sustancialmente más bajo; la curva
  Precision-Recall refleja mejor ese régimen que la ROC [4].

## Interpretabilidad y calibración

- **Grad-CAM** [10] y **Grad-CAM++** [11] generan mapas de calor de las regiones
  más influyentes en cada predicción. Sirven para comprobar si el modelo atiende a
  la lesión y no a artefactos del fondo (pelo, reglas, reflejos). Son una ayuda
  cualitativa, **no una localización del cáncer**.
- **Calibración por *temperature scaling*** [12]: ajusta la confianza del modelo
  para que una probabilidad del 80 % signifique realmente ~80 % de acierto. La
  temperatura `T` de cada modelo se aplica también en el cliente para que la
  confianza mostrada sea honesta.
- **Incertidumbre con *MC Dropout*** [13]: múltiples inferencias con *dropout*
  activo estiman cuánto "duda" el modelo. La demo aún no expone esta señal.

## Arquitectura del sistema

El principio de diseño es mover el cómputo al cliente: la inferencia se ejecuta
en el navegador con TensorFlow.js, sin *backend*. **La imagen del usuario nunca
sale de su dispositivo.**

| Capa | Entorno | Responsabilidad |
|------|---------|-----------------|
| Entrenamiento | Kaggle / Colab (GPU) | Entrenar y exportar los modelos |
| Inferencia | Navegador del usuario | Clasificar con TensorFlow.js |
| Hosting | GitHub Pages + Actions | Servir estáticos y automatizar el deploy |

El modelo `.keras` se convierte a TF.js con cuantización `uint8` (el de por
defecto pesa ~21 MB) y se sirve como estáticos. La app está en React 18 + Vite,
con gestión de memoria de tensores (`tf.tidy` + `dispose`) y validación de la
imagen subida (tipo y tamaño).

## Estructura del repositorio

```
.
├── notebooks/
│   ├── entrenamiento_conjunto_kaggle.ipynb   Los 3 backbones de una tirada (Kaggle)
│   ├── vgg16.ipynb · resnet50v2.ipynb · efficientnetv2s.ipynb
├── demo/                     Demo web (React + TensorFlow.js)
│   ├── src/                  App, inferencia (lib/model.js), Grad-CAM, constantes
│   └── public/model/<id>/    Modelos TF.js (model.json + shards .bin)
├── scripts/
│   ├── convert_kaggle_output.py   .keras → TF.js (local / WSL / Kaggle)
│   ├── kaggle_convert_cell.py     Celda de conversión para Kaggle
│   ├── convert-to-tfjs.mjs        Conversión alternativa con Node
│   ├── gen-notebooks.mjs · gen-og.mjs · download_dataset.ps1
├── models/                   Métricas y notas (los .keras están gitignored)
├── .github/workflows/        ci.yml + deploy.yml (GitHub Pages)
├── requirements.txt · LICENSE · DISCLAIMER.md · CONTRIBUTING.md
```

Excluidos del repositorio (`.gitignore`): `*.keras` (~184 MB), el dataset
(`archive/`) y `node_modules/`.

## Instalación y uso

El repositorio usa **pnpm** para la demo.

### Entrenar

Abre el notebook conjunto en Kaggle con GPU T4 y ejecútalo. Añade el dataset [3]
desde el panel *Data*. Entrena, evalúa (métricas clínicas, calibración,
Grad-CAM) y guarda cada `.keras`.

> Los notebooks se generan con `node scripts/gen-notebooks.mjs` (fuente de
> verdad); no edites los `.ipynb` a mano.

### Convertir los pesos a TF.js

Si el entorno de entrenamiento no tenía `tensorflowjs`, convierte los `.keras`
sin reentrenar:

```bash
python scripts/convert_kaggle_output.py     # Linux / WSL / Kaggle
```

En Windows nativo `tensorflowjs` no instala (por `tensorflow-decision-forests`);
usa WSL o Kaggle. Copia el resultado a `demo/public/model/<id>/`.

### Demo en local

```bash
cd demo
pnpm install
pnpm dev        # http://localhost:5173
pnpm build      # producción
pnpm lint
```

El despliegue a GitHub Pages es automático en cada push a `main` (Actions).

## Limitaciones

- **Sin validación externa.** Un solo conjunto de datos; se desconoce la
  generalización a otros (ISIC, HAM10000 [14]).
- **Métricas medidas en float32; modelo servido cuantizado a uint8.** Las cifras
  de la tabla provienen del modelo `.keras` en precisión completa; la versión
  desplegada en la demo está cuantizada (uint8) para reducir tamaño y no se ha
  reevaluado tras cuantizar. La cuantización puede degradar ligeramente el
  rendimiento real en el navegador.
- **Una sola ejecución**, sin validación cruzada ni intervalos de confianza: las
  diferencias entre modelos deben tomarse con cautela.
- **Dataset balanceado 50/50**, no representativo de la prevalencia real; el
  umbral de decisión no está ajustado a la prevalencia.
- **Posible sesgo demográfico** (fototipos, equipamiento) no caracterizado.
- Solo imágenes **dermatoscópicas**; no aplica a fotos de móvil.

## Consideraciones éticas y clínicas

En detección de cáncer los errores no son simétricos: un falso negativo (cáncer
no detectado) es más grave que un falso positivo (biopsia innecesaria). Por eso
se prioriza la sensibilidad. Aun así, **este proyecto no es un dispositivo médico
ni sustituye el criterio de un profesional sanitario**, y no debe emplearse para
decisiones clínicas. La responsabilidad de cualquier diagnóstico recae en un
dermatólogo certificado.

## Referencias

1. American Cancer Society. *Melanoma Skin Cancer* (información sobre pronóstico y detección temprana). https://www.cancer.org/cancer/types/melanoma-skin-cancer.html
2. Esteva, A., et al. (2017). "Dermatologist-level classification of skin cancer with deep neural networks." *Nature*, 542, 115–118.
3. Javed, H. *Melanoma Skin Cancer Dataset of 10000 Images*. Kaggle (CC0). https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
4. Buda, M., Maki, A., & Mazurowski, M. A. (2018). "A systematic study of the class imbalance problem in convolutional neural networks." *Neural Networks*, 106, 249–259. arXiv:1710.05381
5. Deng, J., et al. (2009). "ImageNet: A Large-Scale Hierarchical Image Database." *CVPR*.
6. Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?" *NeurIPS*. arXiv:1411.1792
7. Tan, M., & Le, Q. (2021). "EfficientNetV2: Smaller Models and Faster Training." *ICML*. arXiv:2104.00298
8. He, K., et al. (2016). "Identity Mappings in Deep Residual Networks." *ECCV*. arXiv:1603.05027
9. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*. arXiv:1409.1556
10. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*. arXiv:1610.02391
11. Chattopadhay, A., et al. (2018). "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks." *WACV*. arXiv:1710.11063
12. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*. arXiv:1706.04599
13. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*. arXiv:1506.02142
14. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). "The HAM10000 dataset." *Scientific Data*, 5, 180161.

*Las métricas de la sección [Resultados](#resultados) proceden de la ejecución
propia descrita en este repositorio, no de las referencias anteriores.*

## Licencia y datos

Código bajo licencia [MIT](LICENSE). El dataset es de dominio público (CC0),
cortesía de Hasnain Javed en Kaggle [3].

## Autor

**Adrián Barriuso Pizarro** — proyecto del Curso de Especialización en IA y Big
Data (IES Ágora, 2024-2025), refactorizado en 2026 para portfolio.

[LinkedIn](https://www.linkedin.com/in/adrian-barriuso/) · [GitHub](https://github.com/abarriuso)
