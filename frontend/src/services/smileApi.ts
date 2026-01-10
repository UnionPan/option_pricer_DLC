import axios from 'axios';
import { VolSmileRequest, VolSmileComparisonResponse } from '../types/smile';
import { API_BASE_URL } from '../config/api.config';

export const smileApi = {
  getVolatilitySmileComparison: async (request: VolSmileRequest): Promise<VolSmileComparisonResponse> => {
    const response = await axios.post(`${API_BASE_URL}/smile/compare`, request);
    return response.data;
  },
};
