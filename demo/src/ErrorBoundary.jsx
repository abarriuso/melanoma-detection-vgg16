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
        <div role="alert" className="error-boundary">
          <p className="error-boundary-title">
            Algo falló al renderizar esta sección
          </p>
          <p className="error-boundary-msg">
            {this.state.error?.message || 'Error desconocido'}
          </p>
          <button
            type="button"
            className="error-boundary-btn"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Reintentar
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
