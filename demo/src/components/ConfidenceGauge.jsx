import { useCountUp } from '../useCountUp';

/**
 * Gauge circular SVG para mostrar confianza.
 * Radio 48, stroke 8px, animado con CSS + contador numérico.
 */
export default function ConfidenceGauge({ confidence, isMalignant, size = 120 }) {
  const animated = useCountUp(confidence * 100, 800);
  const radius = 42;
  const stroke = 7;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const pct = Math.min(Math.max(animated, 0), 100);
  const strokeDashoffset = circumference - (pct / 100) * circumference;
  const color = isMalignant ? 'var(--malignant)' : 'var(--benign)';
  const glowColor = isMalignant ? 'var(--malignant-glow)' : 'var(--benign-glow)';

  return (
    <div
      className="confidence-gauge"
      role="img"
      aria-label={`Confianza: ${animated.toFixed(1)}%`}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="gauge-svg">
        <defs>
          <filter id="gauge-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        <circle
          stroke="var(--white-soft)"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          stroke={color}
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          r={normalizedRadius}
          cx={size / 2}
          cy={size / 2}
          style={{
            strokeDasharray: `${circumference} ${circumference}`,
            strokeDashoffset,
            transform: 'rotate(-90deg)',
            transformOrigin: '50% 50%',
            transition: 'stroke-dashoffset 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
            filter: `drop-shadow(0 0 6px ${glowColor})`,
          }}
        />
      </svg>
      <div className="gauge-text">
        <span className="gauge-value" style={{ color }}>
          {animated.toFixed(1)}%
        </span>
        <span className="gauge-label">confianza</span>
      </div>
    </div>
  );
}
