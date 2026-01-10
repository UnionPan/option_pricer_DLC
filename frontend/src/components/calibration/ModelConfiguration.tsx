import React from 'react';
import './ModelConfiguration.css';
import { CalibrationConfig } from '../../pages/CalibrationPage';

interface ModelConfigurationProps {
  config: CalibrationConfig;
  onConfigChange: (config: Partial<CalibrationConfig>) => void;
  measureType: 'P' | 'Q';
  showDateRange?: boolean;
}

// P-measure models
const P_MEASURE_MODELS = [
  { value: 'gbm', label: 'Geometric Brownian Motion', description: 'μ, σ', method: 'mle' },
  { value: 'ou', label: 'Ornstein-Uhlenbeck', description: 'κ, θ, σ', method: 'mle' },
  { value: 'heston', label: 'Heston (SV)', description: 'κ, θ, ξ, ρ, v₀', method: 'particle_filter' },
  { value: 'rough_bergomi', label: 'Rough Bergomi', description: 'μ, ξ₀, η, ρ, H', method: 'particle_filter' },
  { value: 'regime_switching_gbm', label: 'Regime-Switching GBM', description: 'Multi-regime μ, σ', method: 'em' },
  { value: 'merton_jump', label: 'Merton Jump-Diffusion', description: 'μ, σ, λ, μ_J, σ_J', method: 'mle' },
  { value: 'garch', label: 'GARCH(1,1)', description: 'μ, ω, α, β', method: 'mle' },
];

// Q-measure models
const Q_MEASURE_MODELS = [
  { value: 'heston', label: 'Heston SV (Parametric)', description: 'κ, θ, ξ, ρ, v₀', type: 'Stochastic Vol' },
  { value: 'sabr', label: 'SABR (Parametric)', description: 'α, β, ρ, ν', type: 'Stochastic Vol' },
  { value: 'dupire', label: 'Dupire Local Vol (Non-parametric)', description: 'σ_LV(K,T) surface', type: 'Local Vol' },
];

// P-measure model-specific methods
const P_MEASURE_METHODS: Record<string, Array<{ value: string; label: string }>> = {
  'gbm': [{ value: 'mle', label: 'Maximum Likelihood (Closed-form)' }],
  'ou': [{ value: 'mle', label: 'Maximum Likelihood (Closed-form)' }],
  'heston': [{ value: 'particle_filter', label: 'Particle Filter' }],
  'rough_bergomi': [
    { value: 'moment_matching', label: 'Moment Matching (Variogram)' },
    { value: 'particle_filter', label: 'Particle Filter' },
  ],
  'regime_switching_gbm': [{ value: 'em', label: 'EM Algorithm' }],
  'merton_jump': [{ value: 'mle', label: 'Maximum Likelihood' }],
  'garch': [{ value: 'mle', label: 'Quasi-MLE (Gaussian)' }],
};

// Q-measure model-specific methods
const Q_MEASURE_METHODS: Record<string, Array<{ value: string; label: string }>> = {
  'heston': [
    { value: 'differential_evolution', label: 'Differential Evolution (Global)' },
    { value: 'L-BFGS-B', label: 'L-BFGS-B (Local, Fast)' },
  ],
  'sabr': [
    { value: 'L-BFGS-B', label: 'L-BFGS-B (Fast, Recommended)' },
    { value: 'differential_evolution', label: 'Differential Evolution (Global)' },
  ],
  'dupire': [
    { value: 'extraction', label: 'Direct Extraction (Non-parametric)' },
  ],
};

const ModelConfiguration: React.FC<ModelConfigurationProps> = ({
  config,
  onConfigChange,
  measureType,
  showDateRange = true,
}) => {
  // Set default dates for P-measure (1 year back)
  React.useEffect(() => {
    if (measureType === 'P' && (!config.startDate || !config.endDate)) {
      const end = new Date();
      const start = new Date();
      start.setFullYear(start.getFullYear() - 1);

      onConfigChange({
        startDate: start.toISOString().split('T')[0],
        endDate: end.toISOString().split('T')[0],
      });
    }
  }, [measureType, config.startDate, config.endDate, onConfigChange]);

  // Filter models based on measure type
  const availableModels = measureType === 'Q' ? Q_MEASURE_MODELS : P_MEASURE_MODELS;

  // Get methods for current model
  const availableMethods = React.useMemo(() => {
    if (measureType === 'Q') {
      return Q_MEASURE_METHODS[config.model] || [{ value: 'differential_evolution', label: 'Differential Evolution' }];
    }
    return P_MEASURE_METHODS[config.model] || [{ value: 'mle', label: 'Maximum Likelihood' }];
  }, [config.model, measureType]);

  // Auto-select method when model changes
  React.useEffect(() => {
    if (availableMethods.length > 0) {
      const currentMethod = config.calibrationMethod;
      const validMethods = availableMethods.map(m => m.value);

      // If current method is not valid for this model, select the first available method
      if (!validMethods.includes(currentMethod)) {
        onConfigChange({ calibrationMethod: availableMethods[0].value });
      }
    }
  }, [config.model, measureType, availableMethods]);

  return (
    <div className="model-configuration">
      <h3>Model Configuration</h3>

      <div className="config-row">
        <div className="config-group">
          <label htmlFor="model">Model:</label>
          <select
            id="model"
            value={config.model}
            onChange={(e) => onConfigChange({ model: e.target.value })}
            className="config-select"
          >
            {availableModels.map(model => (
              <option key={model.value} value={model.value}>
                {model.label} ({model.description})
              </option>
            ))}
          </select>
        </div>

        <div className="config-group">
          <label htmlFor="method">Calibration Method:</label>
          <select
            id="method"
            value={config.calibrationMethod}
            onChange={(e) => onConfigChange({ calibrationMethod: e.target.value })}
            className="config-select"
          >
            {availableMethods.map(method => (
              <option key={method.value} value={method.value}>
                {method.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Date Range - Only for P-measure */}
      {measureType === 'P' && showDateRange && (
        <>
          <div className="config-group">
            <label htmlFor="start-date">Start Date:</label>
            <input
              id="start-date"
              type="date"
              value={config.startDate}
              onChange={(e) => onConfigChange({ startDate: e.target.value })}
              className="config-input"
            />
          </div>

          <div className="config-group">
            <label htmlFor="end-date">End Date:</label>
            <input
              id="end-date"
              type="date"
              value={config.endDate}
              onChange={(e) => onConfigChange({ endDate: e.target.value })}
              className="config-input"
            />
          </div>
        </>
      )}

      {/* Risk-free Rate - Only for Q-measure */}
      {measureType === 'Q' && (
        <div className="config-group">
          <label htmlFor="risk-free-rate">Risk-free Rate (r):</label>
          <input
            id="risk-free-rate"
            type="number"
            step="0.001"
            min="0"
            max="1"
            value={config.riskFreeRate || 0.05}
            onChange={(e) => onConfigChange({ riskFreeRate: parseFloat(e.target.value) })}
            className="config-input"
          />
          <p className="helper-text">
            Enter as decimal (e.g., 0.05 for 5%)
          </p>
        </div>
      )}

      {/* Drift Option - Only for P-measure */}
      {measureType === 'P' && (
        <div className="config-group checkbox-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={config.includeDrift}
              onChange={(e) => onConfigChange({ includeDrift: e.target.checked })}
              className="config-checkbox"
            />
            <span>Include Drift Parameter (μ)</span>
          </label>
          <p className="helper-text">
            When unchecked, assumes zero drift (μ = 0)
          </p>
        </div>
      )}

      {/* Q-measure info */}
      {measureType === 'Q' && (
        <div className="config-info">
          <p className="helper-text">
            <strong>Note:</strong> Q-measure calibration uses today's option chain and fixes drift to the risk-free rate (μ = r). Dividend yield is fetched from market data when available.
          </p>

          {/* Model-specific descriptions */}
          {config.model === 'heston' && (
            <p className="helper-text model-description">
              <strong>Heston SV:</strong> Parametric stochastic volatility model. Fast calibration using COS method. Best for capturing volatility smile/skew with mean-reverting volatility.
            </p>
          )}
          {config.model === 'sabr' && (
            <p className="helper-text model-description">
              <strong>SABR:</strong> Industry-standard parametric model using Hagan's analytical formula. Very fast calibration. Popular for interest rate derivatives and FX options. β=0.5 is pre-fixed.
            </p>
          )}
          {config.model === 'dupire' && (
            <p className="helper-text model-description">
              <strong>Dupire Local Vol:</strong> Non-parametric model that extracts local volatility surface σ_LV(K,T) directly from option prices. No optimization needed - uses Dupire's formula with numerical derivatives.
            </p>
          )}
        </div>
      )}

      {/* Quick Date Presets - Only for P-measure */}
      {measureType === 'P' && showDateRange && (
        <div className="date-presets">
          <label>Quick Presets:</label>
          <div className="preset-buttons">
            {[
              { label: '1M', months: 1 },
              { label: '3M', months: 3 },
              { label: '6M', months: 6 },
              { label: '1Y', months: 12 },
              { label: '2Y', months: 24 },
              { label: '5Y', months: 60 },
            ].map(preset => (
              <button
                key={preset.label}
                className="preset-btn"
                onClick={() => {
                  const end = new Date();
                  const start = new Date();
                  start.setMonth(start.getMonth() - preset.months);
                  onConfigChange({
                    startDate: start.toISOString().split('T')[0],
                    endDate: end.toISOString().split('T')[0],
                  });
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelConfiguration;
