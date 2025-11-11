/**
 * Volatility surface types
 */

export interface VolSurfacePoint {
  strike: number;
  expiry: number;
  implied_vol: number;
  moneyness?: number;
}

export interface VolSurfaceRequest {
  symbol: string;
  spot_price?: number;
  min_expiry_days?: number;
  max_expiry_days?: number;
}

export interface VolSurfaceResponse {
  symbol: string;
  spot_price: number;
  surface_points: VolSurfacePoint[];
  num_expirations: number;
  num_strikes: number;
}

export interface VolSmileRequest {
  symbol: string;
  expiration_date: string;
}

export interface VolSmileResponse {
  symbol: string;
  expiration_date: string;
  time_to_expiry: number;
  spot_price: number;
  strikes: number[];
  implied_vols: number[];
}
