/**
 * Pricing and Greeks types
 */

export type OptionType = 'call' | 'put';

export interface PricingRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  volatility: number;
  risk_free_rate: number;
  dividend_yield: number;
  option_type: OptionType;
}

export interface PricingResponse {
  price: number;
  model: string;
}

export interface GreeksRequest {
  spot: number;
  strike: number;
  time_to_expiry: number;
  volatility: number;
  risk_free_rate: number;
  dividend_yield: number;
  option_type: OptionType;
}

export interface GreeksResponse {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}
