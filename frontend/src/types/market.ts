export interface QuoteData {
  symbol: string;
  name: string;
  last_price: number | null;
  change: number | null;
  change_percent: number | null;
  volume: number | null;
}

export interface HistoricalDataPoint {
  date: string;
  close: number;
}

export interface OHLCDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndexChartData {
  symbol: string;
  name: string;
  data: HistoricalDataPoint[];
}

export interface OHLCChartResponse {
  symbol: string;
  data: OHLCDataPoint[];
}

export interface MarketOverviewResponse {
  indices: QuoteData[];
  magnificent7: QuoteData[];
  commodities: QuoteData[];
}

export interface IndexChartsResponse {
  charts: IndexChartData[];
}
