import { useState, useCallback } from 'react';

/**
 * Wrapper seguro de localStorage que no lanza en modo privado
 * o cuando el almacenamiento está deshabilitado.
 */
export function useLocalStorage(key, initialValue) {
  const [stored, setStored] = useState(() => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback(
    (value) => {
      try {
        const next = typeof value === 'function' ? value(stored) : value;
        setStored(next);
        localStorage.setItem(key, JSON.stringify(next));
      } catch {
        // almacenamiento no disponible: la preferencia no persiste
      }
    },
    [key, stored],
  );

  return [stored, setValue];
}
