import React from 'react';
import { QuoteData } from '../../types/market';
import './Magnificent7.css';

interface Magnificent7Props {
  stocks: QuoteData[];
}

const Magnificent7: React.FC<Magnificent7Props> = ({ stocks }) => {
  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A';
    return price.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatChangePercent = (changePercent: number | null) => {
    if (changePercent === null) return 'N/A';
    const sign = changePercent >= 0 ? '+' : '';
    return `${sign}${changePercent.toFixed(2)}%`;
  };

  return (
    <div className="magnificent7-card">
      <h2 className="section-title">Magnificent 7</h2>
      <div className="magnificent7-grid">
        {stocks.map((stock) => (
          <div key={stock.symbol} className="stock-item">
            <div className="stock-header">
              <span className="stock-symbol">{stock.symbol}</span>
              <span className={`stock-change ${stock.change_percent !== null && stock.change_percent >= 0 ? 'positive' : 'negative'}`}>
                {formatChangePercent(stock.change_percent)}
              </span>
            </div>
            <div className="stock-name">{stock.name}</div>
            <div className="stock-price">${formatPrice(stock.last_price)}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Magnificent7;
