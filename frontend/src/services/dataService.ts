/**
 * Market data service
 */
import api from './api';
import { OptionChainResponse, AvailableExpirationsResponse } from '../types/data';

export const dataService = {
  /**
   * Fetch option chain data
   */
  async getOptionChain(symbol: string, expirationDate?: string): Promise<OptionChainResponse> {
    const response = await api.post<OptionChainResponse>('/data/option-chain', {
      symbol,
      expiration_date: expirationDate,
    });
    return response.data;
  },

  /**
   * Get available expiration dates
   */
  async getExpirations(symbol: string): Promise<AvailableExpirationsResponse> {
    const response = await api.get<AvailableExpirationsResponse>(`/data/expirations/${symbol}`);
    return response.data;
  },
};
