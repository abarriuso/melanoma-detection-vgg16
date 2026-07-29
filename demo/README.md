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
conversión en el [README principal](../README.md#b-convertir-keras--tfjs).
