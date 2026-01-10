import React from 'react';
import './CalibrationResults.css';
import { CalibrationResult } from '../../pages/CalibrationPage';

interface CalibrationResultsProps {
  results: CalibrationResult[];
}

const CalibrationResults: React.FC<CalibrationResultsProps> = ({ results }) => {
  const formatNumber = (value: number | null | undefined, decimals: number = 6): string => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return '—';
    }
    return value.toFixed(decimals);
  };

  const getParameterDisplay = (key: string): { name: string; symbol: string; description: string } => {
    const displayInfo: Record<string, { name: string; symbol: string; description: string }> = {
      mu: { name: 'Drift', symbol: 'μ', description: 'Expected return per unit time' },
      sigma: { name: 'Volatility', symbol: 'σ', description: 'Standard deviation of returns' },
      kappa: { name: 'Mean Reversion', symbol: 'κ', description: 'Speed of mean reversion' },
      theta: { name: 'Long-term Mean', symbol: 'θ', description: 'Long-run average level' },
      xi: { name: 'Vol of Vol', symbol: 'ξ', description: 'Volatility of volatility' },
      sigma_v: { name: 'Vol of Vol', symbol: 'σᵥ', description: 'Volatility of variance' },
      rho: { name: 'Correlation', symbol: 'ρ', description: 'Correlation between processes' },
      v0: { name: 'Initial Variance', symbol: 'v₀', description: 'Starting variance level' },
      lambda: { name: 'Jump Intensity', symbol: 'λ', description: 'Average jumps per unit time' },
      mu_j: { name: 'Jump Mean', symbol: 'μⱼ', description: 'Average jump size' },
      sigma_j: { name: 'Jump Volatility', symbol: 'σⱼ', description: 'Jump size volatility' },
      omega: { name: 'GARCH Constant', symbol: 'ω', description: 'Long-term variance level' },
      alpha: { name: 'ARCH Coefficient', symbol: 'α', description: 'Reaction to past shocks' },
      beta: { name: 'GARCH Coefficient', symbol: 'β', description: 'Persistence of volatility' },
      nu: { name: 'Nu', symbol: 'ν', description: 'Parameter nu' },
      xi0: { name: 'Initial Vol', symbol: 'ξ₀', description: 'Initial forward variance' },
      eta: { name: 'Vol of Vol', symbol: 'η', description: 'Volatility of volatility' },
      H: { name: 'Hurst Parameter', symbol: 'H', description: 'Roughness parameter' },
    };
    return displayInfo[key] || { name: key, symbol: key, description: '' };
  };

  const getModelName = (model: string): string => {
    const modelNames: Record<string, string> = {
      gbm: 'Geometric Brownian Motion',
      ou: 'Ornstein-Uhlenbeck',
      heston: 'Heston Stochastic Volatility',
      rough_bergomi: 'Rough Bergomi',
      regime_switching_gbm: 'Regime-Switching GBM',
      merton_jump: 'Merton Jump-Diffusion',
      garch: 'GARCH(1,1)',
    };
    return modelNames[model] || model.toUpperCase();
  };

  return (
    <div className="calibration-results">
      {results.map((result, index) => (
        <div key={index} className="result-card">
          {/* Header */}
          <div className="result-header">
            <div className="header-content">
              <h3>{getModelName(result.model)}</h3>
              <span className="ticker-badge">{result.ticker}</span>
            </div>
            <span className="timestamp">
              {new Date(result.timestamp).toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>

          {/* Parameters Table */}
          <div className="parameters-section">
            <h4>📊 Calibrated Parameters</h4>
            <div className="table-container">
              <table className="parameters-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Symbol</th>
                    <th className="value-column">Value</th>
                    <th className="description-column">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.parameters).map(([key, value]) => {
                    const info = getParameterDisplay(key);
                    return (
                      <tr key={key}>
                        <td className="param-name">{info.name}</td>
                        <td className="param-symbol">{info.symbol}</td>
                        <td className="param-value">{formatNumber(value, 6)}</td>
                        <td className="param-description">{info.description}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Diagnostics Table */}
          {result.diagnostics && Object.keys(result.diagnostics).length > 0 && (
            <div className="diagnostics-section">
              <h4>📈 Model Fit Diagnostics</h4>
              <div className="table-container">
                <table className="diagnostics-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th className="value-column">Value</th>
                      <th className="description-column">Interpretation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.diagnostics.logLikelihood !== undefined && (
                      <tr>
                        <td className="diag-name">Log-Likelihood</td>
                        <td className="diag-value">{formatNumber(result.diagnostics.logLikelihood, 2)}</td>
                        <td className="diag-interpretation">Higher is better (closer to data)</td>
                      </tr>
                    )}
                    {result.diagnostics.aic !== undefined && (
                      <tr>
                        <td className="diag-name">AIC</td>
                        <td className="diag-value">{formatNumber(result.diagnostics.aic, 2)}</td>
                        <td className="diag-interpretation">Lower is better (penalizes complexity)</td>
                      </tr>
                    )}
                    {result.diagnostics.bic !== undefined && (
                      <tr>
                        <td className="diag-name">BIC</td>
                        <td className="diag-value">{formatNumber(result.diagnostics.bic, 2)}</td>
                        <td className="diag-interpretation">Lower is better (stronger penalty)</td>
                      </tr>
                    )}
                    {result.diagnostics.mean_ess !== undefined && (
                      <tr>
                        <td className="diag-name">Mean ESS</td>
                        <td className="diag-value">{formatNumber(result.diagnostics.mean_ess, 0)}</td>
                        <td className="diag-interpretation">Effective sample size (particle filter)</td>
                      </tr>
                    )}
                    {result.diagnostics.n_particles !== undefined && (
                      <tr>
                        <td className="diag-name">Particles</td>
                        <td className="diag-value">{result.diagnostics.n_particles}</td>
                        <td className="diag-interpretation">Number of particles used</td>
                      </tr>
                    )}
                    {result.diagnostics.variogram_r2 !== undefined && (
                      <tr>
                        <td className="diag-name">Variogram R²</td>
                        <td className="diag-value">{formatNumber(result.diagnostics.variogram_r2, 4)}</td>
                        <td className="diag-interpretation">Goodness of fit (0-1, higher better)</td>
                      </tr>
                    )}
                    {result.diagnostics.errorMetrics && Object.entries(result.diagnostics.errorMetrics).map(([key, value]) => (
                      <tr key={key}>
                        <td className="diag-name">{key.replace(/_/g, ' ').toUpperCase()}</td>
                        <td className="diag-value">{formatNumber(value as number, 4)}</td>
                        <td className="diag-interpretation">Error metric</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Note */}
              {result.diagnostics.note && (
                <div className="diagnostic-note">
                  <strong>ℹ️ Note:</strong> {result.diagnostics.note}
                </div>
              )}
            </div>
          )}

          {/* Method Info */}
          <div className="method-info">
            <div className="info-badge">
              <span className="label">Calibration Method:</span>
              <span className="value">{result.method?.toUpperCase() || 'N/A'}</span>
            </div>
            <div className="info-badge">
              <span className="label">Measure:</span>
              <span className="value">{result.measure || 'P-measure'}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default CalibrationResults;
