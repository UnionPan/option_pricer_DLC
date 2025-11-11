/**
 * Market data types
 */

export interface OptionContractData {
  strike: number;
  last_price?: number | null;
  bid?: number | null;
  ask?: number | null;
  volume?: number | null;
  open_interest?: number | null;
  implied_volatility?: number | null;
  option_type: string;
}

export interface OptionChainResponse {
  symbol: string;
  expiration_date: string;
  spot_price: number;
  contracts: OptionContractData[];
  fetched_at: string;
}

export interface AvailableExpirationsResponse {
  symbol: string;
  expirations: string[];
}
