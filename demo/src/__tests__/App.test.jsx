import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock TF.js antes de cualquier import del módulo
vi.mock('@tensorflow/tfjs', () => import('../../__mocks__/@tensorflow/tfjs.js'));

describe('App', () => {
  beforeEach(() => {
    // Mock fetch para evitar errores de red
    globalThis.fetch = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
      })
    );
  });

  it('renderiza el título principal', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    // El título contiene "Melanoma" o "Detección"
    const heading = screen.getByRole('heading', { level: 1 });
    expect(heading).toBeInTheDocument();
    expect(heading.textContent.toLowerCase()).toMatch(/melanoma|detección|lesiones|skin/i);
  });

  it('renderiza el botón de análisis', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    // Debe haber un botón o input para analizar/subir imagen
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('incluye la etiqueta de inferencia local', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    // Debe mencionar que es client-side
    const body = document.body.textContent;
    expect(body.toLowerCase()).toMatch(/navegador|local|client|sin servidor|no se envía/i);
  });
});
