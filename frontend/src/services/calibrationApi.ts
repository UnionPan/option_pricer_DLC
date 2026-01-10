import axios from 'axios';
import { API_BASE_URL } from '../config/api.config';

export interface OHLCVData {
  ticker: string;
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface CalibrationRequest {
  tickers: string[];
  model: string;
  startDate: string;
  endDate: string;
  method: string;
  includeDrift: boolean;
}

export interface CalibrationResponse {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  diagnostics: {
    logLikelihood?: number;
    aic?: number;
    bic?: number;
    mean_ess?: number;
    n_particles?: number;
    variogram_r2?: number;
    errorMetrics?: Record<string, number>;
    note?: string;
  };
  timestamp: string;
  method?: string;
  measure?: string;
}

export interface SimulationRequest {
  model: string;
  parameters: Record<string, number>;
  s0: number;
  nSteps: number;
  nPaths: number;
  dt: number;
  maxPathsReturn?: number;
  randomSeed?: number;
}

export interface SimulationResponse {
  model: string;
  n_steps: number;
  n_paths: number;
  dt: number;
  time_grid: number[];
  mean_path: number[];
  sample_paths: number[][];
  stats: {
    mean: number;
    std: number;
    p5: number;
    p50: number;
    p95: number;
    min: number;
    max: number;
  };
}

// Q-measure interfaces
export interface OptionQuote {
  strike: number;
  expiry: string;
  option_type: string;
  bid: number;
  ask: number;
  mid: number;
  last: number;
  volume: number;
  open_interest: number;
  implied_volatility?: number;
  moneyness: number;
}

export interface OptionChainData {
  ticker: string;
  spot_price: number;
  reference_date: string;
  risk_free_rate: number;
  dividend_yield: number;
  options: OptionQuote[];
  expiries: string[];
  n_options: number;
}

export interface QMeasureCalibrationRequest {
  ticker: string;
  model: string;
  referenceDate?: string;
  riskFreeRate: number;
  filterParams?: {
    min_volume?: number;
    min_open_interest?: number;
    max_spread_pct?: number;
    moneyness_range?: [number, number];
  };
  calibrationMethod: string;
  maxiter: number;
}

export interface QMeasureCalibrationResponse {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  diagnostics: Record<string, any>;
  timestamp: string;
  measure: string;
}

export interface VolSurfaceRequest {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  spotPrice: number;
  riskFreeRate: number;
  dividendYield?: number;
  nStrikes?: number;
  nMaturities?: number;
  strikeRange?: [number, number];
  maturityRange?: [number, number];
}

export interface VolSurfaceResponse {
  strikes: number[];
  maturities: number[];
  vols: number[][];  // 2D array: maturities x strikes
  surface_type: 'implied_vol' | 'local_vol';
  ticker: string;
  model: string;
}

class CalibrationApi {
  /**
   * Fetch OHLCV data from yfinance
   */
  async fetchOHLCV(
    tickers: string[],
    startDate: string,
    endDate: string
  ): Promise<OHLCVData[]> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/fetch-data`, {
        tickers,
        start_date: startDate,
        end_date: endDate,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch OHLCV data');
    }
  }

  /**
   * Run P-measure calibration
   */
  async calibrate(request: CalibrationRequest): Promise<CalibrationResponse[]> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/calibrate`, {
        tickers: request.tickers,
        model: request.model,
        start_date: request.startDate,
        end_date: request.endDate,
        method: request.method,
        include_drift: request.includeDrift,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Calibration failed');
    }
  }

  /**
   * Fetch option chain from yfinance
   */
  async fetchOptionChain(
    ticker: string,
    referenceDate?: string,
    riskFreeRate: number = 0.05,
    expiry?: string
  ): Promise<OptionChainData> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/fetch-option-chain`, {
        ticker,
        reference_date: referenceDate,
        risk_free_rate: riskFreeRate,
        expiry,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to fetch option chain');
    }
  }

  /**
   * Run Q-measure calibration
   */
  async calibrateQMeasure(request: QMeasureCalibrationRequest): Promise<QMeasureCalibrationResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/calibrate-qmeasure`, {
        ticker: request.ticker,
        model: request.model,
        reference_date: request.referenceDate,
        risk_free_rate: request.riskFreeRate,
        filter_params: request.filterParams,
        calibration_method: request.calibrationMethod,
        maxiter: request.maxiter,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Q-measure calibration failed');
    }
  }

  /**
   * Run counterfactual simulation from calibrated parameters
   */
  async simulate(request: SimulationRequest): Promise<SimulationResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/simulate`, {
        model: request.model,
        parameters: request.parameters,
        s0: request.s0,
        n_steps: request.nSteps,
        n_paths: request.nPaths,
        dt: request.dt,
        max_paths_return: request.maxPathsReturn,
        random_seed: request.randomSeed,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Simulation failed');
    }
  }

  /**
   * Generate 3D volatility surface from calibrated Q-measure model
   */
  async generateVolSurface(request: VolSurfaceRequest): Promise<VolSurfaceResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/vol-surface`, {
        ticker: request.ticker,
        model: request.model,
        parameters: request.parameters,
        spot_price: request.spotPrice,
        risk_free_rate: request.riskFreeRate,
        dividend_yield: request.dividendYield,
        n_strikes: request.nStrikes,
        n_maturities: request.nMaturities,
        strike_range: request.strikeRange,
        maturity_range: request.maturityRange,
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Volatility surface generation failed');
    }
  }
}

export const calibrationApi = new CalibrationApi();
