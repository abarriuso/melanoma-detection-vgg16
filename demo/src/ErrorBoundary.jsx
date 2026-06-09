import { Component } from 'react';

/**
 * React Error Boundary: captura errores de rendering en la UI y muestra
 * un fallback en lugar de una pantalla blanca. TF.js crashes, errores
 * de Grad-CAM, etc. no matan toda la app.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          style={{
            padding: '1.5rem',
            margin: '1rem 0',
            border: '1px solid rgba(212, 115, 108, 0.4)',
            background: 'rgba(212, 115, 108, 0.08)',
            color: '#ede8e1',
            fontFamily: 'var(--sans)',
          }}
        >
          <p style={{ margin: '0 0 0.5rem', fontWeight: 600, color: '#d4736c' }}>
            Algo falló al renderizar esta sección
          </p>
          <p style={{ margin: 0, fontSize: '0.85rem', color: '#a09890' }}>
            {this.state.error?.message || 'Error desconocido'}
          </p>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: '0.8rem',
              padding: '0.45rem 0.85rem',
              background: 'transparent',
              border: '1px solid #3d3a37',
              color: '#d4a053',
              cursor: 'pointer',
              fontFamily: 'var(--sans)',
              fontSize: '0.82rem',
              outline: '2px solid #d4a053',
              outlineOffset: '2px',
            }}
          >
            Reintentar
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
