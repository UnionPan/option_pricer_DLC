import React from 'react';
import Plot from 'react-plotly.js';
import { OHLCDataPoint } from '../../types/market';
import { useTheme } from '../../contexts/ThemeContext';
import './OHLCChart.css';

interface OHLCChartProps {
  symbol: string;
  data: OHLCDataPoint[];
}

const OHLCChart: React.FC<OHLCChartProps> = ({ symbol, data }) => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? '#1a1a1a' : '#ffffff';
  const plotTextColor = theme === 'dark' ? '#e5e5e5' : '#1a1a1a';
  const gridColor = theme === 'dark' ? '#333333' : '#e0e0e0';

  return (
    <div className="ohlc-chart-card">
      <h2 className="chart-title">Price History - {symbol}</h2>
      <div className="ohlc-chart-wrapper">
        <Plot
          data={[
            {
              x: data.map((d) => d.date),
              open: data.map((d) => d.open),
              high: data.map((d) => d.high),
              low: data.map((d) => d.low),
              close: data.map((d) => d.close),
              type: 'candlestick',
              increasing: { line: { color: '#22c55e' } },
              decreasing: { line: { color: '#ef4444' } },
              name: symbol,
            } as any,
          ]}
          layout={{
            autosize: true,
            margin: { l: 60, r: 30, t: 10, b: 60 },
            paper_bgcolor: plotBgColor,
            plot_bgcolor: plotBgColor,
            font: { color: plotTextColor, size: 11 },
            xaxis: {
              gridcolor: gridColor,
              showgrid: true,
              rangeslider: { visible: false },
            },
            yaxis: {
              title: { text: 'Price ($)', font: { color: plotTextColor } },
              gridcolor: gridColor,
              showgrid: true,
            },
            hovermode: 'x',
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

export default OHLCChart;
