/**
 * Pricing and Greeks service
 */
import api from './api';
import { PricingRequest, PricingResponse, GreeksRequest, GreeksResponse } from '../types/pricing';

export const pricingService = {
  /**
   * Calculate option price
   */
  async calculatePrice(request: PricingRequest): Promise<PricingResponse> {
    const response = await api.post<PricingResponse>('/pricing/price', request);
    return response.data;
  },

  /**
   * Calculate Greeks
   */
  async calculateGreeks(request: GreeksRequest): Promise<GreeksResponse> {
    const response = await api.post<GreeksResponse>('/pricing/greeks', request);
    return response.data;
  },
};
