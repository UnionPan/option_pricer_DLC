import React, { useState } from 'react';
import { dataService } from '../services/dataService';
import { marketApi } from '../services/marketApi';
import { smileApi } from '../services/smileApi';
import { OptionChainResponse } from '../types/data';
import { OHLCChartResponse } from '../types/market';
import { VolSmileComparisonResponse } from '../types/smile';
import OptionChainTable from '../components/options/OptionChainTable';
import OHLCChart from '../components/charts/OHLCChart';
import VolatilitySmileChart from '../components/charts/VolatilitySmileChart';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import './OptionChainPage.css';

const POPULAR_TICKERS = [
  { value: 'SPY', label: 'SPY - S&P 500 ETF' },
  { value: 'QQQ', label: 'QQQ - NASDAQ 100 ETF' },
  { value: 'AAPL', label: 'AAPL - Apple' },
  { value: 'MSFT', label: 'MSFT - Microsoft' },
  { value: 'GOOGL', label: 'GOOGL - Google' },
  { value: 'AMZN', label: 'AMZN - Amazon' },
  { value: 'NVDA', label: 'NVDA - NVIDIA' },
  { value: 'META', label: 'META - Meta' },
  { value: 'TSLA', label: 'TSLA - Tesla' },
  { value: 'AMD', label: 'AMD - Advanced Micro Devices' },
  { value: 'NFLX', label: 'NFLX - Netflix' },
  { value: 'ORCL', label: 'ORCL - Oracle' },
  { value: 'DIS', label: 'DIS - Disney' },
  { value: 'BA', label: 'BA - Boeing' },
  { value: 'JPM', label: 'JPM - JPMorgan Chase' },
  { value: 'V', label: 'V - Visa' },
  { value: 'WMT', label: 'WMT - Walmart' },
  { value: 'CUSTOM', label: 'Custom Ticker...' },
];

const OptionChainPage: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [customSymbol, setCustomSymbol] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [expirations, setExpirations] = useState<string[]>([]);
  const [selectedExpiration, setSelectedExpiration] = useState<string | null>(null);
  const [chainData, setChainData] = useState<OptionChainResponse | null>(null);
  const [ohlcData, setOhlcData] = useState<OHLCChartResponse | null>(null);
  const [smileData, setSmileData] = useState<VolSmileComparisonResponse | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>(['black_scholes']);
  const [loading, setLoading] = useState(false);
  const [smileLoading, setSmileLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTickerChange = (value: string) => {
    if (value === 'CUSTOM') {
      setShowCustomInput(true);
      setSymbol('');
    } else {
      setShowCustomInput(false);
      setSymbol(value);
    }
    // Reset expirations and chain data when ticker changes
    setExpirations([]);
    setSelectedExpiration(null);
    setChainData(null);
  };

  const handleFetchExpirations = async () => {
    const tickerToFetch = showCustomInput ? customSymbol : symbol;

    if (!tickerToFetch) {
      setError('Please select or enter a symbol');
      return;
    }

    setLoading(true);
    setError(null);
    setExpirations([]);
    setSelectedExpiration(null);
    setChainData(null);

    try {
      const data = await dataService.getExpirations(tickerToFetch.toUpperCase());
      setExpirations(data.expirations);
      if (data.expirations.length > 0) {
        setSelectedExpiration(data.expirations[0]);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch expirations');
    } finally {
      setLoading(false);
    }
  };

  const handleFetchChain = async (expiration: string) => {
    const tickerToFetch = showCustomInput ? customSymbol : symbol;

    setLoading(true);
    setError(null);

    try {
      // Fetch both option chain and OHLC data in parallel
      const [chainResponse, ohlcResponse] = await Promise.all([
        dataService.getOptionChain(tickerToFetch.toUpperCase(), expiration),
        marketApi.getOHLCData(tickerToFetch.toUpperCase(), '6mo'),
      ]);

      setChainData(chainResponse);
      setOhlcData(ohlcResponse);
      setSelectedExpiration(expiration);

      // Also fetch volatility smile for this expiration
      await fetchVolatilitySmile(expiration);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch option chain');
    } finally {
      setLoading(false);
    }
  };

  const fetchVolatilitySmile = async (expiration: string) => {
    const tickerToFetch = showCustomInput ? customSymbol : symbol;

    setSmileLoading(true);

    try {
      const smileResponse = await smileApi.getVolatilitySmileComparison({
        symbol: tickerToFetch.toUpperCase(),
        expiration_date: expiration,
        models: selectedModels,
      });

      setSmileData(smileResponse);
    } catch (err: any) {
      console.error('Failed to fetch volatility smile:', err);
      // Don't show error for smile, just log it
    } finally {
      setSmileLoading(false);
    }
  };

  const handleModelToggle = (model: string) => {
    setSelectedModels((prev) => {
      if (prev.includes(model)) {
        return prev.filter((m) => m !== model);
      } else {
        return [...prev, model];
      }
    });
  };

  return (
    <div className="option-chain-page">
      <h1>Option Chain Viewer</h1>

      <div className="controls">
        <div className="ticker-selector">
          <label htmlFor="ticker-select">Select Ticker:</label>
          <select
            id="ticker-select"
            value={showCustomInput ? 'CUSTOM' : symbol}
            onChange={(e) => handleTickerChange(e.target.value)}
            className="ticker-select"
          >
            {POPULAR_TICKERS.map((ticker) => (
              <option key={ticker.value} value={ticker.value}>
                {ticker.label}
              </option>
            ))}
          </select>
        </div>

        {showCustomInput && (
          <input
            type="text"
            value={customSymbol}
            onChange={(e) => setCustomSymbol(e.target.value)}
            placeholder="Enter custom symbol (e.g., IBM)"
            className="symbol-input"
          />
        )}

        <button onClick={handleFetchExpirations} className="fetch-button">
          Fetch Expirations
        </button>
      </div>

      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}

      {expirations.length > 0 && (
        <>
          <div className="expirations-section">
            <h2>Available Expirations</h2>
            <div className="expiration-buttons">
              {expirations.map((exp) => (
                <button
                  key={exp}
                  onClick={() => handleFetchChain(exp)}
                  className={`expiration-button ${selectedExpiration === exp ? 'selected' : ''}`}
                >
                  {exp}
                </button>
              ))}
            </div>
          </div>

          <div className="models-section">
            <h2>Pricing Models for Volatility Smile</h2>
            <div className="model-checkboxes">
              {[
                { id: 'black_scholes', label: 'Black-Scholes' },
                { id: 'heston', label: 'Heston (Stochastic Vol)' },
                { id: 'sabr', label: 'SABR' },
                { id: 'merton', label: 'Merton (Jump Diffusion)' },
              ].map((model) => (
                <label key={model.id} className="model-checkbox-label">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.id)}
                    onChange={() => handleModelToggle(model.id)}
                    className="model-checkbox"
                  />
                  <span>{model.label}</span>
                </label>
              ))}
            </div>
            {selectedExpiration && (
              <button
                onClick={() => fetchVolatilitySmile(selectedExpiration)}
                className="refresh-smile-button"
                disabled={smileLoading || selectedModels.length === 0}
              >
                {smileLoading ? 'Loading...' : 'Update Volatility Smile'}
              </button>
            )}
          </div>
        </>
      )}

      {chainData && (
        <>
          <div className="chain-info">
            <h2>
              {chainData.symbol} - {chainData.expiration_date}
            </h2>
          </div>
          <OptionChainTable contracts={chainData.contracts} spotPrice={chainData.spot_price} />
        </>
      )}

      {ohlcData && ohlcData.data.length > 0 && (
        <OHLCChart symbol={ohlcData.symbol} data={ohlcData.data} />
      )}

      {smileData && (
        <VolatilitySmileChart data={smileData} />
      )}
    </div>
  );
};

export default OptionChainPage;
