import axios from 'axios';
import { MarketOverviewResponse, IndexChartsResponse, OHLCChartResponse } from '../types/market';
import { API_BASE_URL } from '../config/api.config';

export const marketApi = {
  getMarketOverview: async (): Promise<MarketOverviewResponse> => {
    const response = await axios.get(`${API_BASE_URL}/market/overview`);
    return response.data;
  },

  getIndexCharts: async (): Promise<IndexChartsResponse> => {
    const response = await axios.get(`${API_BASE_URL}/market/charts`);
    return response.data;
  },

  getOHLCData: async (symbol: string, period: string = '6mo'): Promise<OHLCChartResponse> => {
    const response = await axios.get(`${API_BASE_URL}/market/ohlc/${symbol}`, {
      params: { period },
    });
    return response.data;
  },
};
