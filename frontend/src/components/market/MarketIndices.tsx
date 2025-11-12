import React from 'react';
import { QuoteData } from '../../types/market';
import './MarketIndices.css';

interface MarketIndicesProps {
  indices: QuoteData[];
}

const MarketIndices: React.FC<MarketIndicesProps> = ({ indices }) => {
  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A';
    return price.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatChange = (change: number | null, changePercent: number | null) => {
    if (change === null || changePercent === null) return 'N/A';
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`;
  };

  return (
    <div className="market-indices-card">
      <h2 className="section-title">US Equities</h2>
      <div className="indices-list">
        {indices.map((quote) => (
          <div key={quote.symbol} className="index-item">
            <div className="index-header">
              <span className="index-name">{quote.name}</span>
              <span className="index-symbol">{quote.symbol}</span>
            </div>
            <div className="index-data">
              <span className="index-price">${formatPrice(quote.last_price)}</span>
              <span className={`index-change ${quote.change !== null && quote.change >= 0 ? 'positive' : 'negative'}`}>
                {formatChange(quote.change, quote.change_percent)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MarketIndices;
