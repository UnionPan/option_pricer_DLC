import React, { useState } from 'react';
import { surfaceService } from '../services/surfaceService';
import { VolSurfaceResponse } from '../types/surface';
import VolSurface3D from '../components/surface/VolSurface3D';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import './VolSurfacePage.css';

const VolSurfacePage: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [minDays, setMinDays] = useState(7);
  const [maxDays, setMaxDays] = useState(180);
  const [interpolate, setInterpolate] = useState(false);
  const [surfaceData, setSurfaceData] = useState<VolSurfaceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleBuildSurface = async () => {
    if (!symbol) {
      setError('Please enter a symbol');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await surfaceService.buildSurface({
        symbol: symbol.toUpperCase(),
        min_expiry_days: minDays,
        max_expiry_days: maxDays,
        interpolate: interpolate,
        grid_size: 30,
      });
      setSurfaceData(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to build volatility surface');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="vol-surface-page">
      <h1>Volatility Surface</h1>

      <div className="controls">
        <div className="control-group">
          <label>Symbol:</label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="e.g., AAPL"
            className="input-field"
          />
        </div>

        <div className="control-group">
          <label>Min Days to Expiry:</label>
          <input
            type="number"
            value={minDays}
            onChange={(e) => setMinDays(Number(e.target.value))}
            className="input-field"
          />
        </div>

        <div className="control-group">
          <label>Max Days to Expiry:</label>
          <input
            type="number"
            value={maxDays}
            onChange={(e) => setMaxDays(Number(e.target.value))}
            className="input-field"
          />
        </div>

        <div className="control-group">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={interpolate}
              onChange={(e) => setInterpolate(e.target.checked)}
              className="toggle-checkbox"
            />
            <span className="toggle-text">Interpolate Surface</span>
          </label>
        </div>

        <button onClick={handleBuildSurface} className="build-button">
          Build Surface
        </button>
      </div>

      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}

      {surfaceData && (
        <>
          <div className="surface-info">
            <h2>{surfaceData.symbol} Volatility Surface</h2>
            <p>
              <strong>Spot Price:</strong> ${surfaceData.spot_price.toFixed(2)} |{' '}
              <strong>Expirations:</strong> {surfaceData.num_expirations} |{' '}
              <strong>Strikes:</strong> {surfaceData.num_strikes} |{' '}
              <strong>Mode:</strong> {interpolate ? 'Interpolated' : 'Raw Data'}
            </p>
          </div>
          <VolSurface3D surfacePoints={surfaceData.surface_points} symbol={surfaceData.symbol} />
        </>
      )}
    </div>
  );
};

export default VolSurfacePage;
