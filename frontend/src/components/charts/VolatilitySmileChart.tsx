import React from 'react';
import Plot from 'react-plotly.js';
import { VolSmileComparisonResponse } from '../../types/smile';
import { useTheme } from '../../contexts/ThemeContext';
import './VolatilitySmileChart.css';

interface VolatilitySmileChartProps {
  data: VolSmileComparisonResponse;
}

const MODEL_COLORS: { [key: string]: string } = {
  black_scholes: '#667eea',
  heston: '#22c55e',
  sabr: '#f59e0b',
  merton: '#ef4444',
  market_iv: '#8b5cf6',
};

const MODEL_NAMES: { [key: string]: string } = {
  black_scholes: 'Black-Scholes',
  heston: 'Heston',
  sabr: 'SABR',
  merton: 'Merton Jump Diffusion',
  market_iv: 'Market IV',
};

const VolatilitySmileChart: React.FC<VolatilitySmileChartProps> = ({ data }) => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? '#1a1a1a' : '#ffffff';
  const plotTextColor = theme === 'dark' ? '#e5e5e5' : '#1a1a1a';
  const gridColor = theme === 'dark' ? '#333333' : '#e0e0e0';

  // Create traces for market IV and each model
  const traces: any[] = [];

  // Market IV trace
  traces.push({
    x: data.data_points.map((d) => d.moneyness),
    y: data.data_points.map((d) => d.market_iv * 100), // Convert to percentage
    type: 'scatter',
    mode: 'markers+lines',
    name: 'Market IV',
    line: { color: MODEL_COLORS.market_iv, width: 3, dash: 'dot' },
    marker: { size: 8, color: MODEL_COLORS.market_iv },
  });

  // Model calculated IV traces
  data.models_used.forEach((model) => {
    traces.push({
      x: data.data_points.map((d) => d.moneyness),
      y: data.data_points.map((d) => (d.calculated_ivs[model] || 0) * 100),
      type: 'scatter',
      mode: 'lines',
      name: MODEL_NAMES[model] || model,
      line: { color: MODEL_COLORS[model] || '#999999', width: 2 },
    });
  });

  return (
    <div className="vol-smile-chart-card">
      <h2 className="smile-chart-title">
        Volatility Smile - {data.symbol} ({data.expiration_date})
      </h2>
      <div className="smile-info">
        <span><strong>Spot:</strong> ${data.spot_price.toFixed(2)}</span>
        <span><strong>Time to Expiry:</strong> {(data.time_to_expiry * 365).toFixed(0)} days</span>
        <span><strong>Risk-Free Rate:</strong> {(data.risk_free_rate * 100).toFixed(2)}%</span>
      </div>
      <div className="smile-chart-wrapper">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 60, r: 30, t: 10, b: 60 },
            paper_bgcolor: plotBgColor,
            plot_bgcolor: plotBgColor,
            font: { color: plotTextColor, size: 11 },
            xaxis: {
              title: { text: 'Moneyness (K/S)', font: { color: plotTextColor } },
              gridcolor: gridColor,
              showgrid: true,
              zeroline: true,
              zerolinecolor: plotTextColor,
            },
            yaxis: {
              title: { text: 'Implied Volatility (%)', font: { color: plotTextColor } },
              gridcolor: gridColor,
              showgrid: true,
            },
            hovermode: 'x unified',
            legend: {
              x: 1,
              y: 1,
              xanchor: 'right',
              yanchor: 'top',
              bgcolor: 'rgba(0,0,0,0)',
              font: { color: plotTextColor },
            },
          }}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default VolatilitySmileChart;
