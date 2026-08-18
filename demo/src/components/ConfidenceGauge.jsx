/**
 * Gauge circular SVG para mostrar la confianza de la predicción,
 * dibujado como un instrumento de anillo: marcas de dial finas,
 * pista hairline y arco de lectura en el color semántico.
 * El arco se anima vía transición CSS; el valor numérico se muestra
 * directamente (sin contadores animados) para lectura inmediata.
 */
const TICKS = 24;

export default function ConfidenceGauge({ confidence, isMalignant, size = 120 }) {
  const pct = Math.min(Math.max(confidence * 100, 0), 100);
  const radius = 42;
  const stroke = 5;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (pct / 100) * circumference;
  const color = isMalignant ? 'var(--malignant)' : 'var(--benign)';

  // Marcas de dial cada 15°, con las cardinales algo más largas.
  const ticks = Array.from({ length: TICKS }, (_, i) => {
    const angle = (i / TICKS) * 360;
    const major = i % 6 === 0;
    const r1 = 46.5;
    const r2 = major ? 49.5 : 48.3;
    return (
      <line
        key={i}
        x1="50" y1={50 - r1}
        x2="50" y2={50 - r2}
        stroke="var(--hairline-strong)"
        strokeWidth={major ? 0.8 : 0.5}
        transform={`rotate(${angle} 50 50)`}
      />
    );
  });

  return (
    <div
      className="confidence-gauge"
      role="img"
      aria-label={`Confianza: ${pct.toFixed(1)}%`}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} viewBox="0 0 100 100" className="gauge-svg">
        {ticks}
        <circle
          stroke="var(--gauge-track)"
          fill="transparent"
          strokeWidth={1}
          r={45.5}
          cx="50"
          cy="50"
        />
        <circle
          stroke="var(--gauge-track)"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx="50"
          cy="50"
        />
        <circle
          stroke={color}
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          r={normalizedRadius}
          cx="50"
          cy="50"
          style={{
            strokeDasharray: `${circumference} ${circumference}`,
            strokeDashoffset,
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%',
            transition: 'stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1)',
          }}
        />
      </svg>
      <div className="gauge-text">
        <span className="gauge-value" style={{ color }}>
          {pct.toFixed(1)}%
        </span>
        <span className="gauge-label">confianza</span>
      </div>
    </div>
  );
}
