import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import './OHLCVChart.css';
import { Asset } from '../../pages/CalibrationPage';
import { useTheme } from '../../contexts/ThemeContext';

interface OHLCVChartProps {
  data: any[];
  assets: Asset[];
}

const OHLCVChart: React.FC<OHLCVChartProps> = ({ data, assets }) => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? '#0b1220' : '#ffffff';
  const plotTextColor = theme === 'dark' ? '#e5e7eb' : '#0f172a';
  const gridColor = theme === 'dark' ? '#334155' : '#e2e8f0';
  const volumeColor =
    theme === 'dark' ? 'rgba(34, 211, 238, 0.25)' : 'rgba(37, 99, 235, 0.25)';

  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const traces: any[] = [];

    data.forEach((assetData, index) => {
      const asset = assets[index] || { ticker: `Asset ${index + 1}`, name: '' };

      // OHLC Candlestick trace
      traces.push({
        type: 'candlestick',
        x: assetData.dates,
        open: assetData.open,
        high: assetData.high,
        low: assetData.low,
        close: assetData.close,
        name: asset.ticker,
        yaxis: 'y',
        increasing: { line: { color: '#22c55e' } },
        decreasing: { line: { color: '#ef4444' } },
      });

      // Volume bar trace (on secondary y-axis)
      traces.push({
        type: 'bar',
        x: assetData.dates,
        y: assetData.volume,
        name: `${asset.ticker} Volume`,
        yaxis: 'y2',
        marker: {
          color: volumeColor,
        },
        showlegend: false,
      });
    });

    return traces;
  }, [data, assets, volumeColor]);

  const layout: any = {
    title: {
      text: assets.length === 1 ? `${assets[0].ticker} Historical Data` : 'Historical Data',
      font: { size: 16, color: plotTextColor },
    },
    xaxis: {
      title: { text: 'Date', font: { color: plotTextColor } },
      rangeslider: { visible: false },
      type: 'date' as const,
      gridcolor: gridColor,
      zerolinecolor: gridColor,
    },
    yaxis: {
      title: { text: 'Price', font: { color: plotTextColor } },
      domain: [0.25, 1],
      fixedrange: false,
      gridcolor: gridColor,
      zerolinecolor: gridColor,
    },
    yaxis2: {
      title: { text: 'Volume', font: { color: plotTextColor } },
      domain: [0, 0.20],
      fixedrange: false,
      gridcolor: gridColor,
      zerolinecolor: gridColor,
    },
    hovermode: 'x unified' as const,
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      orientation: 'h' as const,
    },
    paper_bgcolor: plotBgColor,
    plot_bgcolor: plotBgColor,
    font: { color: plotTextColor },
    margin: { l: 60, r: 40, t: 60, b: 60 },
  };

  const config: any = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
  };

  if (!data || data.length === 0) {
    return (
      <div className="chart-empty">
        <p>No data to display</p>
      </div>
    );
  }

  return (
    <div className="ohlcv-chart">
      <Plot
        data={chartData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '500px' }}
      />
    </div>
  );
};

export default OHLCVChart;
