import React from 'react';
import { QuoteData } from '../../types/market';
import './CommodityFutures.css';

interface CommodityFuturesProps {
  commodities: QuoteData[];
}

const CommodityFutures: React.FC<CommodityFuturesProps> = ({ commodities }) => {
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
    <div className="commodity-futures-card">
      <h2 className="section-title">Commodity Futures</h2>
      <div className="commodities-grid">
        {commodities.map((commodity) => (
          <div key={commodity.symbol} className="commodity-item">
            <div className="commodity-header">
              <span className="commodity-name">{commodity.name}</span>
              <span className="commodity-symbol">{commodity.symbol}</span>
            </div>
            <div className="commodity-data">
              <div className="commodity-price-section">
                <span className="commodity-label">Last</span>
                <span className="commodity-price">${formatPrice(commodity.last_price)}</span>
              </div>
              <div className="commodity-change-section">
                <span className="commodity-label">Change</span>
                <span className={`commodity-change ${commodity.change !== null && commodity.change >= 0 ? 'positive' : 'negative'}`}>
                  {formatChange(commodity.change, commodity.change_percent)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CommodityFutures;
