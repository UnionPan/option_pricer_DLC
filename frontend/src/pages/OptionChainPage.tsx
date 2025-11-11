import React, { useState } from 'react';
import { dataService } from '../services/dataService';
import { OptionChainResponse } from '../types/data';
import OptionChainTable from '../components/options/OptionChainTable';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import './OptionChainPage.css';

const OptionChainPage: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [chainData, setChainData] = useState<OptionChainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFetchChain = async () => {
    if (!symbol) {
      setError('Please enter a symbol');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await dataService.getOptionChain(symbol.toUpperCase());
      setChainData(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch option chain');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="option-chain-page">
      <h1>Option Chain Viewer</h1>

      <div className="controls">
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="Enter symbol (e.g., AAPL)"
          className="symbol-input"
        />
        <button onClick={handleFetchChain} className="fetch-button">
          Fetch Option Chain
        </button>
      </div>

      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}
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
    </div>
  );
};

export default OptionChainPage;
