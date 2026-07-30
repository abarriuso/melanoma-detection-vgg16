import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

// Mock TF.js antes de cualquier import del módulo
vi.mock('@tensorflow/tfjs', () => import('../../__mocks__/@tensorflow/tfjs.js'));

describe('App', () => {
  beforeEach(() => {
    localStorage.clear();
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

  it('rechaza un archivo de tipo no soportado', async () => {
    const App = (await import('../App.jsx')).default;
    const { container } = render(<App />);
    const input = container.querySelector('input[type="file"]');
    const file = new File(['contenido'], 'lesion.pdf', { type: 'application/pdf' });
    fireEvent.change(input, { target: { files: [file] } });
    expect(await screen.findByRole('alert')).toHaveTextContent(/formato no soportado/i);
  });

  it('rechaza un archivo que supera el tamaño máximo', async () => {
    const App = (await import('../App.jsx')).default;
    const { container } = render(<App />);
    const input = container.querySelector('input[type="file"]');
    const big = new File([new Uint8Array(11 * 1024 * 1024)], 'lesion.jpg', { type: 'image/jpeg' });
    fireEvent.change(input, { target: { files: [big] } });
    expect(await screen.findByRole('alert')).toHaveTextContent(/demasiado grande/i);
  });

  it('deshabilita los modelos sin pesos publicados en el selector', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    const vgg16Radio = screen.getByRole('radio', { name: /vgg16/i });
    const resnetRadio = screen.getByRole('radio', { name: /resnet50v2/i });
    const efficientnetRadio = screen.getByRole('radio', { name: /efficientnetv2s/i });
    expect(vgg16Radio).toBeEnabled();
    expect(resnetRadio).toBeDisabled();
    expect(efficientnetRadio).toBeDisabled();
  });

  it('ignora un modelId guardado de una sesión anterior si el modelo no tiene pesos', async () => {
    localStorage.setItem('modelId', 'resnet50v2');
    const App = (await import('../App.jsx')).default;
    render(<App />);
    const vgg16Radio = screen.getByRole('radio', { name: /vgg16/i });
    expect(vgg16Radio).toBeChecked();
  });
});
