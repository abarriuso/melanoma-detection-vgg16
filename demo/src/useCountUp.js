import { useState, useEffect, useRef } from 'react';

/**
 * Anima un número desde 0 hasta `target` con easing.
 * Devuelve el valor actual (number) para usar directamente en el render.
 */
export function useCountUp(target, duration = 600) {
  const [value, setValue] = useState(0);
  const rafRef = useRef(null);
  const startRef = useRef(null);

  useEffect(() => {
    if (target === 0 || target == null) {
      setValue(0);
      return;
    }
    startRef.current = null;
    const from = 0;
    const to = target;

    const tick = (ts) => {
      if (!startRef.current) startRef.current = ts;
      const elapsed = ts - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      // easeOutExpo
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      setValue(from + (to - from) * ease);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [target, duration]);

  return value;
}
