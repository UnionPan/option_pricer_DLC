import React, { useState } from 'react';
import './AssetSelector.css';
import { Asset } from '../../pages/CalibrationPage';

interface AssetSelectorProps {
  selectedAssets: Asset[];
  onAssetChange: (assets: Asset[]) => void;
}

const POPULAR_ASSETS: Asset[] = [
  { ticker: 'SPY', name: 'S&P 500 ETF' },
  { ticker: 'QQQ', name: 'Nasdaq 100 ETF' },
  { ticker: 'AAPL', name: 'Apple Inc.' },
  { ticker: 'MSFT', name: 'Microsoft' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.' },
  { ticker: 'AMZN', name: 'Amazon' },
  { ticker: 'NVDA', name: 'NVIDIA' },
  { ticker: 'TSLA', name: 'Tesla' },
  { ticker: 'META', name: 'Meta Platforms' },
  { ticker: 'BTC-USD', name: 'Bitcoin' },
  { ticker: 'ETH-USD', name: 'Ethereum' },
];

const AssetSelector: React.FC<AssetSelectorProps> = ({
  selectedAssets,
  onAssetChange,
}) => {
  const [customTicker, setCustomTicker] = useState('');
  const [mode, setMode] = useState<'single' | 'basket'>('single');

  const handleToggleAsset = (asset: Asset) => {
    if (mode === 'single') {
      onAssetChange([asset]);
    } else {
      const isSelected = selectedAssets.some(a => a.ticker === asset.ticker);
      if (isSelected) {
        onAssetChange(selectedAssets.filter(a => a.ticker !== asset.ticker));
      } else {
        onAssetChange([...selectedAssets, asset]);
      }
    }
  };

  const handleAddCustom = () => {
    if (!customTicker.trim()) return;

    const ticker = customTicker.toUpperCase().trim();
    const exists = selectedAssets.some(a => a.ticker === ticker);

    if (!exists) {
      const newAsset: Asset = { ticker, name: ticker };
      if (mode === 'single') {
        onAssetChange([newAsset]);
      } else {
        onAssetChange([...selectedAssets, newAsset]);
      }
    }
    setCustomTicker('');
  };

  const handleModeChange = (newMode: 'single' | 'basket') => {
    setMode(newMode);
    if (newMode === 'single' && selectedAssets.length > 1) {
      onAssetChange([selectedAssets[0]]);
    }
  };

  const isAssetSelected = (ticker: string) => {
    return selectedAssets.some(a => a.ticker === ticker);
  };

  return (
    <div className="asset-selector">
      <h3>Asset Selection</h3>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
          onClick={() => handleModeChange('single')}
        >
          Single Asset
        </button>
        <button
          className={`mode-btn ${mode === 'basket' ? 'active' : ''}`}
          onClick={() => handleModeChange('basket')}
        >
          Basket
        </button>
      </div>

      {/* Selected Assets Display */}
      {selectedAssets.length > 0 && (
        <div className="selected-assets">
          <label>Selected ({selectedAssets.length}):</label>
          <div className="asset-chips">
            {selectedAssets.map(asset => (
              <div key={asset.ticker} className="asset-chip">
                {asset.ticker}
                <button
                  className="remove-btn"
                  onClick={() => handleToggleAsset(asset)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Popular Assets Grid */}
      <div className="popular-assets">
        <label>Popular Assets:</label>
        <div className="asset-grid">
          {POPULAR_ASSETS.map(asset => (
            <button
              key={asset.ticker}
              className={`asset-btn ${isAssetSelected(asset.ticker) ? 'selected' : ''}`}
              onClick={() => handleToggleAsset(asset)}
              title={asset.name}
            >
              {asset.ticker}
            </button>
          ))}
        </div>
      </div>

      {/* Custom Ticker Input */}
      <div className="custom-input">
        <label>Custom Ticker:</label>
        <div className="input-group">
          <input
            type="text"
            value={customTicker}
            onChange={(e) => setCustomTicker(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddCustom()}
            placeholder="Enter ticker (e.g., AAPL)"
            className="ticker-input"
          />
          <button
            className="add-btn"
            onClick={handleAddCustom}
            disabled={!customTicker.trim()}
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
};

export default AssetSelector;
