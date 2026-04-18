import React from 'react';
import Plot from 'react-plotly.js';
import { IndexChartData } from '../../types/market';
import { useTheme } from '../../contexts/ThemeContext';
import './IndexCharts.css';

interface IndexChartsProps {
  charts: IndexChartData[];
}

const IndexCharts: React.FC<IndexChartsProps> = ({ charts }) => {
  const { theme } = useTheme();

  const plotBgColor = 'rgba(0,0,0,0)';
  const plotTextColor = theme === 'dark' ? '#94a3b8' : '#475569';
  const gridColor = theme === 'dark' ? 'rgba(148,163,184,0.08)' : 'rgba(15,23,42,0.06)';

  // Define colors for each index
  const indexColors = {
    'S&P 500': '#06b6d4',
    'NASDAQ': '#818cf8',
  };

  // Normalize data to percentage change from first data point
  const normalizeToPercentage = (data: { date: string; close: number }[]) => {
    if (data.length === 0) return [];
    const baseValue = data[0].close;
    return data.map((d) => ({
      date: d.date,
      percentChange: ((d.close - baseValue) / baseValue) * 100,
    }));
  };

  // Create traces for all charts on the same plot
  const traces: any[] = charts.map((chart) => {
    const normalizedData = normalizeToPercentage(chart.data);
    return {
      x: normalizedData.map((d) => d.date),
      y: normalizedData.map((d) => d.percentChange),
      type: 'scatter',
      mode: 'lines',
      name: chart.name,
      line: {
        color: indexColors[chart.name as keyof typeof indexColors] || '#667eea',
        width: 2
      },
    };
  });

  return (
    <div className="index-charts-card">
      <h2 className="section-title">Index Performance (1 Year)</h2>
      <div className="combined-chart-wrapper">
        <Plot
          data={traces}
          layout={{
            autosize: true,
            margin: { l: 60, r: 30, t: 20, b: 40 },
            paper_bgcolor: plotBgColor,
            plot_bgcolor: plotBgColor,
            font: { color: plotTextColor, size: 11 },
            xaxis: {
              gridcolor: gridColor,
              showgrid: true,
              zeroline: false,
            },
            yaxis: {
              title: { text: 'Change (%)', font: { color: plotTextColor } },
              ticksuffix: '%',
              gridcolor: gridColor,
              showgrid: true,
              zeroline: true,
              zerolinecolor: plotTextColor,
              zerolinewidth: 1,
            },
            hovermode: 'x unified',
            legend: {
              x: 0.5,
              y: 1.1,
              xanchor: 'center',
              yanchor: 'bottom',
              orientation: 'h',
              font: { color: plotTextColor },
            },
          }}
          config={{
            responsive: true,
            displayModeBar: false,
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

export default IndexCharts;
