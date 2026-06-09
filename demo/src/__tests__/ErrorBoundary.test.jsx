import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from '../ErrorBoundary';

function ThrowingComponent() {
  throw new Error('Test error');
}

function GoodComponent() {
  return <div>Child content</div>;
}

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <GoodComponent />
      </ErrorBoundary>
    );
    expect(screen.getByText('Child content')).toBeInTheDocument();
  });

  it('renders fallback UI on error', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    );
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/algo falló/i)).toBeInTheDocument();
    expect(screen.getByText('Test error')).toBeInTheDocument();
    spy.mockRestore();
  });

  it('shows retry button that resets error state', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    let shouldThrow = true;
    function ConditionalThrow() {
      if (shouldThrow) throw new Error('Boom');
      return <div>Recovered</div>;
    }
    const { rerender } = render(
      <ErrorBoundary>
        <ConditionalThrow />
      </ErrorBoundary>
    );
    expect(screen.getByText('Reintentar')).toBeInTheDocument();
    shouldThrow = false;
    fireEvent.click(screen.getByText('Reintentar'));
    rerender(
      <ErrorBoundary>
        <ConditionalThrow />
      </ErrorBoundary>
    );
    expect(screen.getByText('Recovered')).toBeInTheDocument();
    spy.mockRestore();
  });
});
