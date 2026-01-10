import axios from 'axios';
import { API_BASE_URL } from '../config/api.config';

export interface EnvironmentConfig {
  s0: number;
  strike: number;
  maturity: number;
  volatility: number;
  risk_free_rate: number;
  n_steps: number;
  transaction_cost_bps: number;
}

export interface RLInferenceRequest {
  agent_type: 'ppo' | 'sac' | 'td3' | 'deep_hedging' | 'ais_hedging';
  environment_config: EnvironmentConfig;
  use_demo_model: boolean;
  model_path?: string;
  random_seed?: number;
}

export interface RLInferenceResponse {
  agent_type: string;
  time_grid: number[];
  spot_path: number[];
  hedge_positions: number[];
  pnl: number[];
  final_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  num_rebalances: number;
  total_transaction_costs: number;
  environment_config: EnvironmentConfig;
}

export interface AgentInfo {
  type: string;
  name: string;
  description: string;
  features: string[];
  available: boolean;
}

class RLHedgingApi {
  /**
   * Run inference with a pre-trained RL agent
   */
  async runInference(request: RLInferenceRequest): Promise<RLInferenceResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/rl-hedging/inference`, request);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'RL inference failed');
    }
  }

  /**
   * Get list of available RL agents
   */
  async listAgents(): Promise<AgentInfo[]> {
    try {
      const response = await axios.get(`${API_BASE_URL}/rl-hedging/agents`);
      return response.data.agents;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch agents');
    }
  }
}

export const rlHedgingApi = new RLHedgingApi();
