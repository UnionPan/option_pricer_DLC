import axios from 'axios';
import { VolSmileRequest, VolSmileComparisonResponse } from '../types/smile';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const smileApi = {
  getVolatilitySmileComparison: async (request: VolSmileRequest): Promise<VolSmileComparisonResponse> => {
    const response = await axios.post(`${API_BASE_URL}/smile/compare`, request);
    return response.data;
  },
};
