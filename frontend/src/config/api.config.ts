/**
 * API Configuration
 *
 * The API base URL is determined by the environment:
 * - Development: Uses REACT_APP_API_URL from .env.development (defaults to localhost:8000)
 * - Production: Uses REACT_APP_API_URL from .env.production
 * - Can be overridden with .env.local for local development
 */

export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const API_CONFIG = {
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
};

// Log current API URL in development
if (process.env.NODE_ENV === 'development') {
  console.log('[API Config] Using API URL:', API_BASE_URL);
}
