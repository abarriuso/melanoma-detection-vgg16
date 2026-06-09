import { describe, it, expect } from 'vitest';
import { UMBRAL, DATASET_NAME, DATASET_URL, GITHUB_USER, REPO_NAME } from '../constants';

describe('constants', () => {
  it('UMBRAL is 0.5', () => {
    expect(UMBRAL).toBe(0.5);
  });

  it('DATASET_NAME is a non-empty string', () => {
    expect(typeof DATASET_NAME).toBe('string');
    expect(DATASET_NAME.length).toBeGreaterThan(0);
  });

  it('DATASET_URL is a valid URL', () => {
    expect(DATASET_URL).toMatch(/^https?:\/\//);
  });

  it('GITHUB_USER defaults to abarriuso', () => {
    expect(GITHUB_USER).toBe('abarriuso');
  });

  it('REPO_NAME defaults to melanoma-detection-vgg16', () => {
    expect(REPO_NAME).toBe('melanoma-detection-vgg16');
  });
});
