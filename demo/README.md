# Demo web — Desarrollo

Referencia rápida para desarrollo local. La documentación completa del proyecto
(arquitectura, seguridad, conversión del modelo, despliegue) está en el
[README principal](../README.md).

## Desarrollo local

```bash
cd demo
pnpm install
pnpm dev      # http://localhost:5173
pnpm build    # producción → demo/dist/
pnpm lint     # ESLint
```

El modelo convertido debe estar en `public/model/` — ver instrucciones de
conversión en el [README principal](../README.md#convertir-los-pesos-a-tfjs).

## Limitaciones de cabeceras en GitHub Pages

GitHub Pages **no permite enviar cabeceras HTTP propias**, solo sirve
estáticos. Las cabeceras de seguridad que no podemos desplegar y su riesgo
residual:

| Cabecera no disponible | Riesgo residual | Mitigación aplicada |
|---|---|---|
| `Content-Security-Policy` (cabecera) | Una CSP por `<meta http-equiv>` no admite `frame-ancestors` | CSP completa por `<meta>` (la inyecta `vite.config.js` como primer elemento de `<head>`) |
| `frame-ancestors` / `X-Frame-Options` | Clickjacking: la página podría incrustarse en un iframe | `public/frame-guard.js`: oculta el documento y rompe el iframe; sin JS la app no funciona en absoluto |
| `X-Content-Type-Options: nosniff` | MIME-sniffing en navegadores antiguos | Riesgo bajo: todo el contenido se sirve con los tipos correctos de Pages |
| `Referrer-Policy` | La URL (sin datos sensibles) puede viajar en `Referer` a terceros | No hay enlaces salientes con datos; los externos usan `rel="noreferrer"` |
| `Permissions-Policy` | APIs del navegador (cámara, geolocalización…) no restringidas por cabecera | La app no las usa; la CSP limita lo que puede cargarse |

Si el proyecto madura y se necesitan estas cabeceras de verdad, habría que
migrar el hosting a uno que las permita (Cloudflare Pages, Netlify).

## Imágenes de ejemplo (`public/samples/`)

Las 120 miniaturas de `public/samples/{benign,malignant}/` son imágenes
dermatoscópicas reales procedentes de dos datasets de Kaggle publicados con
licencia **CC0** (dominio público), según declaran sus páginas de dataset:

- [Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) (Hasnain Javed)
- [Skin Lesion Analysis Toward Melanoma Detection (ISIC 2017)](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection) (wanderdust)

Se usan solo como ejemplos de prueba de la demo; el modelo no se entrena con
el segundo dataset (ver `scripts/download_examples_dataset.ps1` y
`scripts/refresh-samples.mjs`). Las imágenes dermatoscópicas no muestran
rasgos faciales identificables.

**Criterio de retirada:** si un titular de derechos, un paciente o el autor
de un dataset solicita la retirada de alguna imagen —o si la licencia CC0
declarada resultara incorrecta—, se eliminará de inmediato. Para solicitarlo,
abre un *issue* en el
[repositorio](https://github.com/abarriuso/melanoma-detection-vgg16/issues)
indicando el nombre del fichero; no hace falta justificar el motivo.
