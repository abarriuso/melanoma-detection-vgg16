import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ResultCard from '../components/ResultCard';

const result = {
  score: 0.87,
  logit: 1.9,
  ms: 120,
  label: 'Maligno',
  confidence: 0.87,
  esMaligno: true,
};

describe('ResultCard', () => {
  it('anuncia el resultado a lectores de pantalla: región con aria-live="polite" (F-03)', () => {
    render(<ResultCard result={result} />);
    const region = screen.getByRole('region', { name: /resultado del análisis/i });
    expect(region).toHaveAttribute('aria-live', 'polite');
    expect(region).toHaveTextContent(/maligno/i);
    expect(region).toHaveTextContent(/87/);
  });

  it('no renderiza nada sin resultado', () => {
    const { container } = render(<ResultCard result={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
