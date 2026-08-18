import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

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

  it('renderiza el selector de imagen', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    // Debe haber un botón o input para subir imagen
    const buttons = screen.getAllByRole('button');
    expect(buttons.length).toBeGreaterThan(0);
  });

  it('no hay botón manual de "Analizar imagen": el análisis es automático', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    expect(screen.queryByRole('button', { name: /^analizar imagen$/i })).not.toBeInTheDocument();
  });

  it('sube un archivo válido sin necesitar ningún clic adicional', async () => {
    const App = (await import('../App.jsx')).default;
    const { container } = render(<App />);
    const input = container.querySelector('input[type="file"]');
    const file = new File([new Uint8Array(100)], 'lesion.jpg', { type: 'image/jpeg' });
    fireEvent.change(input, { target: { files: [file] } });
    // Se decodifica de forma async (new Image()); solo comprobamos que no
    // aparece ningún error de tipo/tamaño, es decir, el archivo se acepta.
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByText(/formato no soportado|demasiado/i)).not.toBeInTheDocument();
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

  it('habilita los tres modelos en el selector (todos con pesos publicados)', async () => {
    const App = (await import('../App.jsx')).default;
    render(<App />);
    // findByRole: re-query + timeout explícito de 3 s. La cadena async de
    // carga del modelo (backend → pesos → warmup) son 6+ microtasks y con
    // CPU contendida (CI) el timeout por defecto de 1 s se agotaba de forma
    // intermitente.
    const vgg16Radio = await screen.findByRole('radio', { name: /vgg16/i }, { timeout: 3000 });
    const resnetRadio = await screen.findByRole('radio', { name: /resnet50v2/i }, { timeout: 3000 });
    const efficientnetRadio = await screen.findByRole('radio', { name: /efficientnetv2s/i }, { timeout: 3000 });
    // Los radios se habilitan cuando el modelo termina de cargar (async).
    await waitFor(() => {
      expect(vgg16Radio).toBeEnabled();
      expect(resnetRadio).toBeEnabled();
      expect(efficientnetRadio).toBeEnabled();
    }, { timeout: 3000 });
  });

  it('usa el modelo por defecto (EfficientNetV2S) si el modelId guardado no existe', async () => {
    localStorage.setItem('modelId', 'modelo_inexistente');
    const App = (await import('../App.jsx')).default;
    render(<App />);
    const efficientnetRadio = screen.getByRole('radio', { name: /efficientnetv2s/i });
    expect(efficientnetRadio).toBeChecked();
  });

  it('Enter sobre un enlace enfocado no dispara el análisis ni cancela la navegación (F-02)', async () => {
    // jsdom no decodifica imágenes ni soporta blob URLs: se simulan para
    // que handleFile complete y la app llegue al estado "imagen cargada".
    const RealImage = globalThis.Image;
    const realCreate = URL.createObjectURL;
    const realRevoke = URL.revokeObjectURL;
    globalThis.Image = class {
      constructor() {
        this.naturalWidth = 224;
        this.naturalHeight = 224;
      }
      set src(_) {
        setTimeout(() => this.onload?.(), 0);
      }
    };
    URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    URL.revokeObjectURL = vi.fn();
    try {
      const App = (await import('../App.jsx')).default;
      const { container } = render(<App />);
      const input = container.querySelector('input[type="file"]');
      const file = new File([new Uint8Array(100)], 'lesion.jpg', { type: 'image/jpeg' });
      fireEvent.change(input, { target: { files: [file] } });
      // Esperar a que el preview aparezca = imageURL cargada, atajo armado.
      await screen.findByAltText(/lesión dermatoscópica/i, {}, { timeout: 3000 });

      // Enter sobre un enlace: el atajo global debe ignorarlo.
      // (Nombre exacto: el footer tiene otro enlace "Código fuente".)
      const link = screen.getByRole('link', { name: 'Código' });
      link.focus();
      const linkDefaultAllowed = fireEvent.keyDown(link, { key: 'Enter' });
      expect(linkDefaultAllowed).toBe(true); // sin preventDefault: navegación intacta

      // Enter sobre un role="button" sin handler propio: igualmente ignorado
      // (el dropzone real sí tiene handler propio y hace su preventDefault
      // legítimo para abrir el selector de archivos).
      const dummy = document.createElement('div');
      dummy.setAttribute('role', 'button');
      document.body.appendChild(dummy);
      const roleButtonDefaultAllowed = fireEvent.keyDown(dummy, { key: 'Enter' });
      expect(roleButtonDefaultAllowed).toBe(true);
      dummy.remove();

      // Control: con el foco fuera de interactivos el atajo sigue armado.
      const bodyDefaultAllowed = fireEvent.keyDown(document.body, { key: 'Enter' });
      expect(bodyDefaultAllowed).toBe(false); // preventDefault aplicado
    } finally {
      globalThis.Image = RealImage;
      URL.createObjectURL = realCreate;
      URL.revokeObjectURL = realRevoke;
    }
  });
});
