export interface VolSmileDataPoint {
  strike: number;
  moneyness: number;
  market_iv: number;
  calculated_ivs: { [model: string]: number };
}

export interface VolSmileComparisonResponse {
  symbol: string;
  expiration_date: string;
  time_to_expiry: number;
  spot_price: number;
  risk_free_rate: number;
  data_points: VolSmileDataPoint[];
  models_used: string[];
}

export interface VolSmileRequest {
  symbol: string;
  expiration_date: string;
  models?: string[];
  // Heston parameters
  heston_v0?: number;
  heston_theta?: number;
  heston_kappa?: number;
  heston_sigma_v?: number;
  heston_rho?: number;
  // SABR parameters
  sabr_alpha?: number;
  sabr_beta?: number;
  sabr_rho?: number;
  sabr_nu?: number;
  // Merton parameters
  merton_lambda?: number;
  merton_mu_j?: number;
  merton_sigma_j?: number;
}
