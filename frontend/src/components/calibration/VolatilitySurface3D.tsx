import React from 'react';
import Plot from 'react-plotly.js';
import './VolatilitySurface3D.css';
import { OptionChainData } from '../../services/calibrationApi';
import { useTheme } from '../../contexts/ThemeContext';

interface VolSurfaceData {
  strikes: number[];
  maturities: number[];
  vols: number[][];  // 2D array: maturities x strikes
  surface_type: 'implied_vol' | 'local_vol';
  ticker: string;
  model: string;
}

interface VolatilitySurface3DProps {
  surfaceData: VolSurfaceData | null;
  isLoading: boolean;
  optionChainData?: OptionChainData | null;
  spotPrice?: number;
}

const VolatilitySurface3D: React.FC<VolatilitySurface3DProps> = ({
  surfaceData,
  isLoading,
  optionChainData,
  spotPrice,
}) => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? 'rgba(15, 23, 42, 0.9)' : 'rgba(248, 250, 252, 0.9)';
  const plotTextColor = theme === 'dark' ? '#e5e7eb' : '#0f172a';
  const gridColor =
    theme === 'dark' ? 'rgba(148, 163, 184, 0.2)' : 'rgba(100, 116, 139, 0.2)';

  if (isLoading) {
    return (
      <div className="vol-surface-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Generating volatility surface...</p>
        </div>
      </div>
    );
  }

  if (!surfaceData) {
    return (
      <div className="vol-surface-container">
        <div className="vol-surface-placeholder">
          <p>📊 Volatility surface will appear here after calibration</p>
          <p className="helper-text">
            The surface shows how volatility varies across strikes and maturities
          </p>
        </div>
      </div>
    );
  }

  const { strikes, maturities, vols, surface_type, ticker, model } = surfaceData;

  // Convert strikes to moneyness for better visualization
  const spot = spotPrice || optionChainData?.spot_price || 100;
  const moneyness = strikes.map(k => k / spot);

  // Create mesh grid for 3D surface
  const data: any = [
    {
      type: 'surface',
      x: moneyness,  // Moneyness (K/S)
      y: maturities,  // Time to maturity
      z: vols,  // Volatility values
      colorscale: [
        [0, '#0d0887'],      // Deep purple (low vol)
        [0.25, '#5302a3'],   // Purple
        [0.5, '#8b0aa5'],    // Magenta
        [0.75, '#db3a07'],   // Red-orange
        [1, '#f98e09'],      // Orange (high vol)
      ],
      colorbar: {
        title: {
          text: surface_type === 'implied_vol' ? 'Implied Vol' : 'Local Vol',
          side: 'right',
          font: { color: plotTextColor },
        },
        tickfont: { color: plotTextColor },
        tickformat: '.1%',
        len: 0.7,
      },
      hovertemplate:
        '<b>Moneyness:</b> %{x:.3f}<br>' +
        '<b>Maturity:</b> %{y:.2f} years<br>' +
        '<b>Vol:</b> %{z:.2%}<extra></extra>',
    },
  ];

  const layout: any = {
    title: {
      text: `${ticker} - ${model.toUpperCase()} ${
        surface_type === 'implied_vol' ? 'Implied' : 'Local'
      } Volatility Surface`,
      font: { size: 16 },
    },
    scene: {
      xaxis: {
        title: 'Moneyness (K/S)',
        gridcolor: gridColor,
        tickfont: { color: plotTextColor },
      },
      yaxis: {
        title: 'Time to Maturity (years)',
        gridcolor: gridColor,
        tickfont: { color: plotTextColor },
      },
      zaxis: {
        title: surface_type === 'implied_vol' ? 'Implied Volatility' : 'Local Volatility',
        tickformat: '.1%',
        gridcolor: gridColor,
        tickfont: { color: plotTextColor },
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.3 },
      },
      bgcolor: plotBgColor,
    },
    autosize: true,
    margin: { l: 0, r: 0, t: 40, b: 0 },
    paper_bgcolor: 'transparent',
    font: { color: plotTextColor },
  };

  const config: any = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    displaylogo: false,
  };

  return (
    <div className="vol-surface-container">
      <div className="surface-info">
        <div className="info-item">
          <span className="label">Model:</span>
          <span className="value">{model.toUpperCase()}</span>
        </div>
        <div className="info-item">
          <span className="label">Surface Type:</span>
          <span className="value">
            {surface_type === 'implied_vol' ? 'Implied Volatility' : 'Local Volatility'}
          </span>
        </div>
        <div className="info-item">
          <span className="label">Grid:</span>
          <span className="value">
            {strikes.length} strikes × {maturities.length} maturities
          </span>
        </div>
      </div>

      <div className="vol-surface-plot">
        <Plot
          data={data}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '600px' }}
        />
      </div>

      <div className="surface-description">
        {surface_type === 'implied_vol' ? (
          <p>
            <strong>Implied Volatility Surface:</strong> Shows the market-implied volatility
            extracted from option prices across moneyness and time to maturity. The smile/skew
            pattern indicates how volatility varies across strikes and expiration dates.
          </p>
        ) : (
          <p>
            <strong>Local Volatility Surface:</strong> Shows the instantaneous volatility
            σ_LV(K,T) at each strike and maturity. This is the input to the local volatility
            model and is calibrated to match market option prices exactly.
          </p>
        )}
      </div>
    </div>
  );
};

export default VolatilitySurface3D;
