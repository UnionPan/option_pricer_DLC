/**
 * Volatility surface service
 */
import api from './api';
import { VolSurfaceRequest, VolSurfaceResponse, VolSmileRequest, VolSmileResponse } from '../types/surface';

export const surfaceService = {
  /**
   * Build volatility surface
   */
  async buildSurface(request: VolSurfaceRequest): Promise<VolSurfaceResponse> {
    const response = await api.post<VolSurfaceResponse>('/surface/build', request);
    return response.data;
  },

  /**
   * Get volatility smile for specific expiration
   */
  async getSmile(request: VolSmileRequest): Promise<VolSmileResponse> {
    const response = await api.post<VolSmileResponse>('/surface/smile', request);
    return response.data;
  },
};
