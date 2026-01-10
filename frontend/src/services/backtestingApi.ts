import axios from 'axios';
import { API_BASE_URL } from '../config/api.config';

export interface LiabilitySpec {
  option_type: string;
  strike: number;
  maturity_days: number;
  quantity: number;
}

export interface HedgeOptionSpec {
  option_type: string;
  strike: number;
  maturity_days: number;
}

export interface BacktestRequest {
  model: string;
  parameters: Record<string, number>;
  liability_spec: LiabilitySpec;
  hedging_strategy: string;
  hedge_options?: HedgeOptionSpec[];
  heston_pricer?: string;
  s0: number;
  n_steps: number;
  n_paths: number;
  dt: number;
  risk_free_rate: number;
  transaction_cost_bps: number;
  rebalance_threshold: number;
  random_seed?: number;
  full_visualization?: boolean;
}

export interface Greeks {
  delta: number[];
  gamma: number[];
  vega: number[];
  theta: number[];
}

export interface Transaction {
  time: number;
  spot: number;
  action: string;
  shares_traded?: number;
  contracts_traded?: number;
  option_strike?: number;
  option_type?: string;
  hedge_leg?: number;
  cost: number;
  delta?: number;
  transaction_cost?: number;
}

export interface SummaryStats {
  mean_pnl: number;
  std_pnl: number;
  median_pnl: number;
  min_pnl: number;
  max_pnl: number;
  sharpe_ratio: number;
  var_95: number;
  cvar_95: number;
  num_rebalances: number;
  total_transaction_costs: number;
}

export interface OptionData {
  strike: number;
  maturity_days: number;
  tau_remaining: number;
  option_type: string;
  price: number;
  moneyness: number;
  implied_volatility: number;
}

export interface OptionChainData {
  time_step: number;
  time: number;
  spot: number;
  volatility: number;
  options: OptionData[];
}

export interface IVSurfaceData {
  moneyness: number[];
  ttm: number[];
  iv: number[];
  option_type: string[];
  time_step: number[];
  spot: number[];
}

export interface BacktestResponse {
  time_grid: number[];
  representative_path: number[];
  all_paths: number[][];
  variance_path: number[] | null;
  volatility_path: number[];
  hedge_positions: number[];
  hedge_option_positions?: number[][] | null;
  hedge_option_value?: number[][] | null;
  cash: number[];
  portfolio_value: number[];
  option_value: number[];
  pnl: number[];
  greeks: Greeks;
  transactions: Transaction[];
  summary_stats: SummaryStats;
  final_pnl_distribution: number[];
  option_chains?: OptionChainData[] | null;
  iv_surface?: IVSurfaceData | null;
  liability_spec: LiabilitySpec;
  hedge_option_specs?: HedgeOptionSpec[] | null;
  hedging_strategy: string;
  model: string;
  parameters: Record<string, number>;
}

class BacktestingApi {
  /**
   * Run backtesting simulation
   */
  async runBacktest(request: BacktestRequest): Promise<BacktestResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/backtesting/run`, request);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Backtesting failed');
    }
  }
}

export const backtestingApi = new BacktestingApi();
