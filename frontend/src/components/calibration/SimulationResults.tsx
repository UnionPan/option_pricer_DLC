import React from 'react';
import Plot from 'react-plotly.js';
import './SimulationResults.css';
import { SimulationResponse } from '../../services/calibrationApi';
import { useTheme } from '../../contexts/ThemeContext';

interface SimulationResultsProps {
  data: SimulationResponse;
}

const SimulationResults: React.FC<SimulationResultsProps> = ({ data }) => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? '#0b1220' : '#ffffff';
  const plotTextColor = theme === 'dark' ? '#e5e7eb' : '#0f172a';
  const gridColor = theme === 'dark' ? '#334155' : '#e2e8f0';
  const traceColor = theme === 'dark' ? 'rgba(34, 211, 238, 0.25)' : 'rgba(37, 99, 235, 0.25)';
  const meanColor = theme === 'dark' ? '#22d3ee' : '#2563eb';

  const traces: any[] = data.sample_paths.map((path) => ({
    x: data.time_grid,
    y: path,
    type: 'scatter',
    mode: 'lines',
    line: { width: 1, color: traceColor },
    showlegend: false,
    hoverinfo: 'skip',
  }));

  traces.push({
    x: data.time_grid,
    y: data.mean_path,
    type: 'scatter',
    mode: 'lines',
    name: 'Mean Path',
    line: { width: 2.5, color: meanColor },
  });

  return (
    <div className="simulation-results">
      <div className="simulation-chart">
        <h2>Counterfactual Simulation</h2>
        <Plot
          data={traces as any}
          layout={{
            title: { text: 'Spot Paths' },
            xaxis: { title: { text: 'Time (years)' }, gridcolor: gridColor },
            yaxis: { title: { text: 'Spot Price' }, gridcolor: gridColor },
            paper_bgcolor: plotBgColor,
            plot_bgcolor: plotBgColor,
            font: { color: plotTextColor },
            margin: { t: 40, r: 20, l: 50, b: 40 },
            height: 320,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      <div className="simulation-stats">
        <h3>Final Spot Statistics</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Mean</span>
            <span className="stat-value">{data.stats.mean.toFixed(4)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Std</span>
            <span className="stat-value">{data.stats.std.toFixed(4)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">P5</span>
            <span className="stat-value">{data.stats.p5.toFixed(4)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Median</span>
            <span className="stat-value">{data.stats.p50.toFixed(4)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">P95</span>
            <span className="stat-value">{data.stats.p95.toFixed(4)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Min / Max</span>
            <span className="stat-value">
              {data.stats.min.toFixed(4)} / {data.stats.max.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationResults;
