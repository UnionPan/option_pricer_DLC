/**
 * API client for Options Desk backend
 */
import axios from 'axios';
import { API_CONFIG } from '../config/api.config';

const api = axios.create(API_CONFIG);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default api;
