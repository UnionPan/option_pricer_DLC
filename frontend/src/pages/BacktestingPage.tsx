import React, { useState } from 'react';
import './BacktestingPage.css';
import { backtestingApi, BacktestRequest, BacktestResponse } from '../services/backtestingApi';
import { CalibrationResult } from './CalibrationPage';
import { DEFAULT_CALIBRATIONS, MODEL_DESCRIPTIONS } from '../data/defaultCalibrations';
import Plot from 'react-plotly.js';
import { useTheme } from '../contexts/ThemeContext';

type HedgeOptionSpec = {
  option_type: 'call' | 'put';
  strike: number;
  maturity_days: number;
};

const BacktestingPage: React.FC = () => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? '#0b1220' : '#ffffff';
  const plotTextColor = theme === 'dark' ? '#e5e7eb' : '#0f172a';
  const gridColor = theme === 'dark' ? '#334155' : '#e2e8f0';
  const axisTitle = (text: string) => ({ text, font: { color: plotTextColor } });

  // Configuration state
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResult | null>(null);
  const [liabilityConfig, setLiabilityConfig] = useState({
    optionType: 'call' as 'call' | 'put',
    strike: 100,
    maturityDays: 30,
    quantity: -1.0,
  });
  const [hedgingStrategy, setHedgingStrategy] = useState('delta_hedge');
  const [hedgeOptions, setHedgeOptions] = useState<HedgeOptionSpec[]>(() => ([
    { option_type: 'call', strike: 105, maturity_days: 30 },
    { option_type: 'call', strike: 110, maturity_days: 30 },
  ]));
  const [syncHedgeToLiability, setSyncHedgeToLiability] = useState(true);
  const [hestonPricer, setHestonPricer] = useState<'mgf' | 'analytical'>('mgf');
  const [simulationConfig, setSimulationConfig] = useState({
    s0: 100,
    nSteps: 63, // ~3 months of daily steps (63 trading days)
    nPaths: 100,
    dt: 1 / 252,
    riskFreeRate: 0.05,
    transactionCostBps: 5.0,
    rebalanceThreshold: 0.05,
  });

  // Results state
  const [backtestResult, setBacktestResult] = useState<BacktestResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Animation state
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.2);

  // Visualization toggle
  const [fullVisualization, setFullVisualization] = useState(true);

  React.useEffect(() => {
    if (!syncHedgeToLiability) return;
    setHedgeOptions([
      { option_type: 'call', strike: Number((liabilityConfig.strike * 1.05).toFixed(2)), maturity_days: liabilityConfig.maturityDays },
      { option_type: 'call', strike: Number((liabilityConfig.strike * 1.1).toFixed(2)), maturity_days: liabilityConfig.maturityDays },
    ]);
  }, [liabilityConfig.strike, liabilityConfig.maturityDays, syncHedgeToLiability]);

  const requiredHedgeLegs =
    hedgingStrategy === 'delta_gamma_vega_hedge'
      ? 2
      : hedgingStrategy === 'delta_gamma_hedge' || hedgingStrategy === 'delta_vega_hedge'
        ? 1
        : 0;

  // Handle file upload for calibration results
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string);

        // Handle both single result and array
        const result = Array.isArray(json) ? json[0] : json;

        setCalibrationResult(result);

        // Auto-populate simulation config
        if (json.ticker) {
          setSimulationConfig(prev => ({
            ...prev,
            s0: 100, // Default, user can adjust
          }));
        }

        setError(null);
      } catch (err) {
        setError('Failed to parse calibration results JSON');
      }
    };
    reader.readAsText(file);
  };

  const handleRunBacktest = async () => {
    if (!calibrationResult) {
      setError('Please upload calibration results first');
      return;
    }

    setIsRunning(true);
    setError(null);
    setCurrentStep(0);
    setIsPlaying(false);

    try {
      const request: BacktestRequest = {
        model: calibrationResult.model,
        parameters: calibrationResult.parameters,
        liability_spec: {
          option_type: liabilityConfig.optionType,
          strike: liabilityConfig.strike,
          maturity_days: liabilityConfig.maturityDays,
          quantity: liabilityConfig.quantity,
        },
        hedging_strategy: hedgingStrategy,
        hedge_options: requiredHedgeLegs > 0 ? hedgeOptions.slice(0, requiredHedgeLegs) : undefined,
        heston_pricer: hestonPricer,
        s0: simulationConfig.s0,
        n_steps: simulationConfig.nSteps,
        n_paths: fullVisualization ? 1 : simulationConfig.nPaths,
        dt: simulationConfig.dt,
        risk_free_rate: simulationConfig.riskFreeRate,
        transaction_cost_bps: simulationConfig.transactionCostBps,
        rebalance_threshold: simulationConfig.rebalanceThreshold,
        full_visualization: fullVisualization,
      };

      const result = await backtestingApi.runBacktest(request);
      setBacktestResult(result);
    } catch (err: any) {
      setError(err.message || 'Backtesting failed');
    } finally {
      setIsRunning(false);
    }
  };

  // Animation control
  React.useEffect(() => {
    if (!isPlaying || !backtestResult) return;

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        const next = prev + 1;
        if (next >= backtestResult.time_grid.length) {
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, 200 / playbackSpeed); // 200ms base speed (slower)

    return () => clearInterval(interval);
  }, [isPlaying, backtestResult, playbackSpeed]);

  const togglePlayback = () => {
    if (currentStep >= (backtestResult?.time_grid.length || 0) - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const resetAnimation = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  // Prepare visualization data
  const getAnimatedData = () => {
    if (!backtestResult) return null;

    const upToStep = currentStep + 1;
    return {
      timeGrid: backtestResult.time_grid.slice(0, upToStep),
      path: backtestResult.representative_path.slice(0, upToStep),
      hedge: backtestResult.hedge_positions.slice(0, upToStep),
      pnl: backtestResult.pnl.slice(0, upToStep),
      optionValue: backtestResult.option_value.slice(0, upToStep),
      greeks: {
        delta: backtestResult.greeks.delta.slice(0, upToStep),
        gamma: backtestResult.greeks.gamma.slice(0, upToStep),
        vega: backtestResult.greeks.vega.slice(0, upToStep),
      },
    };
  };

  const animatedData = getAnimatedData();

  return (
    <div className="backtesting-page">
      <div className="page-header">
        <h1>Hedging Strategy Backtesting</h1>
        <p className="page-subtitle">
          Simulate option hedging with calibrated P-measure models
        </p>
      </div>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
          <button className="error-close" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="backtesting-layout">
        {/* Configuration Panel - Horizontal Layout */}
        <div className="config-panel">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
            {/* Model Selection - Top Left */}
            <div className="config-section" style={{ margin: 0 }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Model</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <select
                  className="model-select"
                  style={{ width: '100%' }}
                  value={calibrationResult ? 'custom' : ''}
                  onChange={(e) => {
                    if (e.target.value === 'custom') return;
                    const modelKey = e.target.value as keyof typeof DEFAULT_CALIBRATIONS;
                    if (modelKey && DEFAULT_CALIBRATIONS[modelKey]) {
                      setCalibrationResult(DEFAULT_CALIBRATIONS[modelKey] as any);
                    }
                  }}
                >
                  <option value="">Choose...</option>
                  {Object.entries(MODEL_DESCRIPTIONS).map(([key, desc]) => (
                    <option key={key} value={key}>
                      {desc.name}
                    </option>
                  ))}
                  {calibrationResult && <option value="custom">Custom: {calibrationResult.ticker}</option>}
                </select>
                <div className="file-upload">
                  <label className="upload-label" style={{ fontSize: '0.75rem' }}>
                    <input
                      type="file"
                      accept=".json"
                      onChange={handleFileUpload}
                      style={{ display: 'none' }}
                    />
                    <span className="upload-btn" style={{ padding: '0.375rem 0.75rem', fontSize: '0.75rem' }}>Upload JSON</span>
                  </label>
                </div>
                {calibrationResult && (
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                    {calibrationResult.model.toUpperCase()} • {calibrationResult.ticker}
                  </div>
                )}
              </div>
            </div>

            {/* Liability Specification - Middle */}
            <div className="config-section" style={{ margin: 0 }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Liability</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>Type</label>
                  <select
                    value={liabilityConfig.optionType}
                    onChange={(e) => setLiabilityConfig({
                      ...liabilityConfig,
                      optionType: e.target.value as 'call' | 'put'
                    })}
                    style={{ fontSize: '0.875rem' }}
                  >
                    <option value="call">Call</option>
                    <option value="put">Put</option>
                  </select>
                </div>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>Strike</label>
                  <input
                    type="number"
                    value={liabilityConfig.strike}
                    onChange={(e) => setLiabilityConfig({
                      ...liabilityConfig,
                      strike: parseFloat(e.target.value)
                    })}
                    style={{ fontSize: '0.875rem' }}
                  />
                </div>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>Maturity (d)</label>
                  <input
                    type="number"
                    value={liabilityConfig.maturityDays}
                    onChange={(e) => setLiabilityConfig({
                      ...liabilityConfig,
                      maturityDays: parseInt(e.target.value)
                    })}
                    style={{ fontSize: '0.875rem' }}
                  />
                </div>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>Quantity</label>
                  <input
                    type="number"
                    step="0.1"
                    value={liabilityConfig.quantity}
                    onChange={(e) => setLiabilityConfig({
                      ...liabilityConfig,
                      quantity: parseFloat(e.target.value)
                    })}
                    style={{ fontSize: '0.875rem' }}
                  />
                </div>
              </div>
            </div>

            {/* Hedging Strategy - Right */}
            <div className="config-section" style={{ margin: 0 }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Strategy</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {[
                  { value: 'delta_hedge', label: 'Delta' },
                  { value: 'delta_gamma_hedge', label: 'Δ-Γ' },
                  { value: 'delta_vega_hedge', label: 'Δ-ν' },
                  { value: 'delta_gamma_vega_hedge', label: 'Δ-Γ-ν' },
                ].map((strategy) => (
                  <label
                    key={strategy.value}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.5rem',
                      border: '2px solid var(--border-color)',
                      borderRadius: '0.25rem',
                      cursor: 'pointer',
                      fontSize: '0.875rem',
                      fontWeight: hedgingStrategy === strategy.value ? 600 : 500,
                      backgroundColor: hedgingStrategy === strategy.value ? 'rgba(37, 99, 235, 0.15)' : 'transparent',
                      borderColor: hedgingStrategy === strategy.value ? 'var(--accent)' : 'var(--border-color)',
                      color: hedgingStrategy === strategy.value ? 'var(--accent)' : 'var(--text-primary)',
                    }}
                  >
                    <input
                      type="radio"
                      name="strategy"
                      value={strategy.value}
                      checked={hedgingStrategy === strategy.value}
                      onChange={(e) => setHedgingStrategy(e.target.value)}
                      style={{ margin: 0 }}
                    />
                    <span>{strategy.label}</span>
                  </label>
                ))}
              </div>
              {requiredHedgeLegs > 0 && (
                <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Hedge Options
                  </div>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                    <input
                      type="checkbox"
                      checked={syncHedgeToLiability}
                      onChange={(e) => setSyncHedgeToLiability(e.target.checked)}
                      style={{ width: 'auto', margin: 0 }}
                    />
                    <span>Sync to liability</span>
                  </label>
                  {hedgeOptions.slice(0, requiredHedgeLegs).map((opt, idx) => (
                    <div key={`hedge-${idx}`} style={{ display: 'grid', gridTemplateColumns: '0.9fr 1fr 1fr', gap: '0.4rem' }}>
                      <select
                        value={opt.option_type}
                        onChange={(e) => {
                          const next = [...hedgeOptions];
                          next[idx] = { ...next[idx], option_type: e.target.value as 'call' | 'put' };
                          setHedgeOptions(next);
                          setSyncHedgeToLiability(false);
                        }}
                        style={{ fontSize: '0.8rem' }}
                      >
                        <option value="call">Call</option>
                        <option value="put">Put</option>
                      </select>
                      <input
                        type="number"
                        value={opt.strike}
                        onChange={(e) => {
                          const next = [...hedgeOptions];
                          next[idx] = { ...next[idx], strike: parseFloat(e.target.value) };
                          setHedgeOptions(next);
                          setSyncHedgeToLiability(false);
                        }}
                        style={{ fontSize: '0.8rem' }}
                        placeholder="Strike"
                      />
                      <input
                        type="number"
                        value={opt.maturity_days}
                        onChange={(e) => {
                          const next = [...hedgeOptions];
                          next[idx] = { ...next[idx], maturity_days: parseInt(e.target.value) };
                          setHedgeOptions(next);
                          setSyncHedgeToLiability(false);
                        }}
                        style={{ fontSize: '0.8rem' }}
                        placeholder="Days"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Simulation Parameters - Top Right */}
            <div className="config-section" style={{ margin: 0 }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem' }}>Simulation</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>S0</label>
                  <input
                    type="number"
                    value={simulationConfig.s0}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      s0: parseFloat(e.target.value)
                    })}
                    style={{ fontSize: '0.875rem' }}
                  />
                </div>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>Steps</label>
                  <input
                    type="number"
                    value={simulationConfig.nSteps}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      nSteps: parseInt(e.target.value)
                    })}
                    style={{ fontSize: '0.875rem' }}
                  />
                </div>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>Paths</label>
                  <input
                    type="number"
                    value={fullVisualization ? 1 : simulationConfig.nPaths}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      nPaths: parseInt(e.target.value)
                    })}
                    disabled={fullVisualization}
                    style={{ fontSize: '0.875rem', opacity: fullVisualization ? 0.6 : 1 }}
                  />
                </div>
                <div className="config-field">
                  <label style={{ fontSize: '0.75rem' }}>TC (bps)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={simulationConfig.transactionCostBps}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      transactionCostBps: parseFloat(e.target.value)
                    })}
                    style={{ fontSize: '0.875rem' }}
                  />
                </div>
              </div>
              {calibrationResult?.model?.toLowerCase().includes('heston') && (
                <div className="config-field" style={{ marginTop: '0.5rem' }}>
                  <label style={{ fontSize: '0.75rem' }}>Heston Pricer</label>
                  <select
                    value={hestonPricer}
                    onChange={(e) => setHestonPricer(e.target.value as 'mgf' | 'analytical')}
                    style={{ fontSize: '0.875rem' }}
                  >
                    <option value="mgf">MGF (fast)</option>
                    <option value="analytical">Analytical (slow)</option>
                  </select>
                </div>
              )}
              <div className="config-field" style={{ marginTop: '0.5rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', fontSize: '0.75rem' }}>
                  <input
                    type="checkbox"
                    checked={fullVisualization}
                    onChange={(e) => setFullVisualization(e.target.checked)}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Single path mode</span>
                </label>
                <p style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '0.25rem', marginLeft: '1.25rem' }}>
                  {fullVisualization
                    ? '1 path with IV surface visualization'
                    : 'Multi-path for summary statistics'}
                </p>
              </div>
            </div>
          </div>

          <button
            className="btn btn-primary btn-large"
            onClick={handleRunBacktest}
            disabled={isRunning || !calibrationResult}
          >
            {isRunning ? 'Running Backtest...' : 'Run Backtest'}
          </button>
        </div>

        {/* Results Panel */}
        {backtestResult && (
          <div className="results-panel" key={`results-${backtestResult.time_grid.length}`}>
            {/* Animation Controls (only in single-path mode) */}
            {fullVisualization && (
            <div className="animation-controls">
              <button onClick={togglePlayback} className="control-btn">
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
              <button onClick={resetAnimation} className="control-btn">
                ⏮ Reset
              </button>
              <div className="time-display">
                Step: {currentStep} / {backtestResult.time_grid.length - 1}
                {' '}({(backtestResult.time_grid[currentStep] * 365).toFixed(0)} days)
              </div>
              <div className="speed-control">
                <label>Speed:</label>
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
                />
                <span>{playbackSpeed.toFixed(1)}x</span>
              </div>
            </div>
            )}

            {/* Charts Layout */}
            {fullVisualization && animatedData && (
              <div style={{ display: 'flex', gap: '1rem' }}>
                {/* Left: Vertical Stack of Main Charts (only in single-path mode) */}
                <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {/* Chart 1: Price + Volatility (dual y-axis) */}
                  <div className="chart-container">
                    <Plot
                      key="price-vol-chart"
                      data={[
                        {
                          x: animatedData.timeGrid.map(t => t * 365),
                          y: animatedData.path,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'Spot Price',
                          line: { color: '#2E86AB', width: 2 },
                          yaxis: 'y',
                        },
                        {
                          x: [animatedData.timeGrid[currentStep] * 365],
                          y: [animatedData.path[currentStep]],
                          type: 'scatter',
                          mode: 'markers',
                          name: 'Current',
                          marker: { color: '#ff6b6b', size: 10 },
                          yaxis: 'y',
                          showlegend: false,
                        },
                        ...(backtestResult.volatility_path ? [{
                          x: backtestResult.time_grid.map(t => t * 365),
                          y: backtestResult.volatility_path.slice(0, currentStep + 1).map(v => v * 100),
                          type: 'scatter' as const,
                          mode: 'lines' as const,
                          name: 'Volatility',
                          line: { color: '#A23B72', width: 2, dash: 'dot' as any },
                          yaxis: 'y2',
                        }] : []),
                      ]}
                      layout={{
                        title: { text: 'Price & Volatility Dynamics', font: { size: 14 } },
                        datarevision: currentStep,
                        xaxis: { title: axisTitle('Time (days)'), gridcolor: gridColor },
                        yaxis: {
                          title: axisTitle('Spot Price ($)'),
                          gridcolor: gridColor,
                          side: 'left',
                        },
                        yaxis2: {
                          title: axisTitle('Vol (%)'),
                          overlaying: 'y',
                          side: 'right',
                          gridcolor: 'transparent',
                        },
                        legend: { x: 0.02, y: 0.98, orientation: 'v', font: { size: 10 } },
                        paper_bgcolor: plotBgColor,
                        plot_bgcolor: plotBgColor,
                        font: { color: plotTextColor },
                        height: 340,
                        margin: { l: 50, r: 50, t: 35, b: 45 },
                      }}
                      config={{ responsive: true, displayModeBar: false, staticPlot: false }}
                      style={{ width: '100%' }}
                    />
                  </div>

                  {/* Chart 2: P&L + Delta + Gamma + Vega */}
                  <div className="chart-container">
                    <Plot
                      key="pnl-greeks-chart"
                      data={[
                        {
                          x: animatedData.timeGrid.map(t => t * 365),
                          y: animatedData.pnl,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'P&L',
                          line: { color: animatedData.pnl[currentStep] >= 0 ? '#4CAF50' : '#f44336', width: 2 },
                          yaxis: 'y',
                        },
                        {
                          x: animatedData.timeGrid.map(t => t * 365),
                          y: animatedData.greeks.delta,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'Delta',
                          line: { color: '#9C27B0', width: 1.5 },
                          yaxis: 'y2',
                        },
                        {
                          x: animatedData.timeGrid.map(t => t * 365),
                          y: animatedData.greeks.gamma,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'Gamma',
                          line: { color: '#00BCD4', width: 1.5 },
                          yaxis: 'y2',
                        },
                        {
                          x: animatedData.timeGrid.map(t => t * 365),
                          y: animatedData.greeks.vega,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'Vega',
                          line: { color: '#FF9800', width: 1.5, dash: 'dot' as any },
                          yaxis: 'y2',
                        },
                      ]}
                      layout={{
                        title: { text: 'Portfolio P&L & Greeks', font: { size: 14 } },
                        datarevision: currentStep,
                        xaxis: { title: axisTitle('Time (days)'), gridcolor: gridColor },
                        yaxis: {
                          title: axisTitle('P&L ($)'),
                          gridcolor: gridColor,
                          side: 'left',
                        },
                        yaxis2: {
                          title: axisTitle('Greeks'),
                          overlaying: 'y',
                          side: 'right',
                          gridcolor: 'transparent',
                        },
                        legend: { x: 0.02, y: 0.98, orientation: 'v', font: { size: 9 } },
                        paper_bgcolor: plotBgColor,
                        plot_bgcolor: plotBgColor,
                        font: { color: plotTextColor },
                        height: 340,
                        margin: { l: 50, r: 50, t: 35, b: 45 },
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%' }}
                    />
                  </div>

                </div>

                {/* Right: IV Surface + Vol Smile (only in single-path mode) */}
                {fullVisualization && (
                <div style={{ width: '550px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

                  {/* Debug message if option chains not available */}
                  {(!backtestResult.option_chains || backtestResult.option_chains.length === 0) && (
                    <div style={{
                      padding: '2rem',
                      textAlign: 'center',
                      color: 'var(--text-secondary)',
                      fontSize: '0.875rem'
                    }}>
                      Loading option chain visualization...
                      {backtestResult.option_chains === undefined && ' (chains undefined)'}
                      {backtestResult.option_chains != null && backtestResult.option_chains.length === 0 && ' (chains empty)'}
                    </div>
                  )}

                  {/* IV Surface - Dynamic with current timestep */}
                  {backtestResult.option_chains && backtestResult.option_chains.length > 0 && (() => {
                    const currentChain = backtestResult.option_chains[Math.min(currentStep, backtestResult.option_chains.length - 1)];
                    const callOptions = currentChain.options.filter((opt: any) => opt.option_type === 'call');

                    // Create interpolated surface grid
                    // Group by maturity, sort by moneyness
                    const byMaturity: Record<number, Array<{m: number, iv: number}>> = {};
                    callOptions.forEach((opt: any) => {
                      if (!byMaturity[opt.maturity_days]) {
                        byMaturity[opt.maturity_days] = [];
                      }
                      byMaturity[opt.maturity_days].push({
                        m: opt.moneyness,
                        iv: opt.implied_volatility * 100
                      });
                    });

                    // Create grid for surface plot
                    const maturities = Object.keys(byMaturity).map(Number).sort((a, b) => a - b);
                    const zData: number[][] = [];
                    const yData: number[] = [];
                    let xData: number[] = [];

                    maturities.forEach(ttm => {
                      const points = byMaturity[ttm].sort((a, b) => a.m - b.m);
                      if (xData.length === 0) {
                        xData = points.map(p => p.m);
                      }
                      zData.push(points.map(p => p.iv));
                      yData.push(ttm);
                    });

                    return (
                      <div className="chart-container">
                        <Plot
                          key="iv-surface-chart"
                          data={[
                            {
                              type: 'surface',
                              x: xData,
                              y: yData,
                              z: zData,
                              colorscale: 'Viridis',
                              showscale: true,
                              colorbar: {
                                title: { text: 'IV%', font: { color: plotTextColor, size: 10 } },
                                tickfont: { color: plotTextColor, size: 10 },
                                len: 0.5,
                              },
                              hovertemplate: 'K/S: %{x:.2f}<br>TTM: %{y}d<br>IV: %{z:.1f}%<extra></extra>',
                              contours: {
                                z: {
                                  show: true,
                                  usecolormap: true,
                                  highlightcolor: "#42f462",
                                  project: { z: true }
                                }
                              },
                            },
                          ]}
                          layout={{
                            title: { text: `IV Surface (t=${currentStep})`, font: { size: 12 } },
                            datarevision: currentStep,
                            scene: {
                              xaxis: { title: { text: 'K/S', font: { size: 10 } }, tickfont: { size: 9 } },
                              yaxis: { title: { text: 'TTM', font: { size: 10 } }, tickfont: { size: 9 } },
                              zaxis: { title: { text: 'IV%', font: { size: 10 } }, tickfont: { size: 9 } },
                              camera: { eye: { x: 1.3, y: 1.3, z: 1.2 } },
                              bgcolor: plotBgColor,
                            },
                            paper_bgcolor: plotBgColor,
                            font: { color: plotTextColor },
                            height: 400,
                            margin: { l: 0, r: 0, t: 30, b: 0 },
                          }}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      </div>
                    );
                  })()}

                  {/* Vol Smile/Skew */}
                  {backtestResult.option_chains && backtestResult.option_chains.length > 0 && (() => {
                    const currentChain = backtestResult.option_chains[Math.min(currentStep, backtestResult.option_chains.length - 1)];
                    const byMaturity: Record<number, { moneyness: number[], iv: number[] }> = {};
                    currentChain.options.forEach((opt: any) => {
                      if (opt.option_type === 'call') {
                        if (!byMaturity[opt.maturity_days]) {
                          byMaturity[opt.maturity_days] = { moneyness: [], iv: [] };
                        }
                        byMaturity[opt.maturity_days].moneyness.push(opt.moneyness);
                        byMaturity[opt.maturity_days].iv.push(opt.implied_volatility * 100);
                      }
                    });

                    const traces = Object.entries(byMaturity).map(([ttm, data], idx) => {
                      const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
                      const sorted = data.moneyness.map((m, i) => ({ m, iv: data.iv[i] })).sort((a, b) => a.m - b.m);
                      return {
                        x: sorted.map(d => d.m),
                        y: sorted.map(d => d.iv),
                        type: 'scatter' as const,
                        mode: 'lines+markers' as const,
                        name: `${ttm}d`,
                        line: { color: colors[idx % colors.length], width: 2 },
                        marker: { size: 4 },
                      };
                    });

                    return (
                      <div className="chart-container">
                        <Plot
                          key="vol-smile-chart"
                          data={traces}
                          layout={{
                            title: { text: 'Vol Smile', font: { size: 12 } },
                            datarevision: currentStep,
                            xaxis: { title: { text: 'K/S', font: { size: 10 } }, gridcolor: gridColor },
                            yaxis: { title: { text: 'IV%', font: { size: 10 } }, gridcolor: gridColor },
                            legend: { font: { size: 9 }, orientation: 'h', y: -0.2 },
                            paper_bgcolor: plotBgColor,
                            plot_bgcolor: plotBgColor,
                            font: { color: plotTextColor },
                            height: 280,
                            margin: { l: 50, r: 20, t: 30, b: 60 },
                            shapes: [{
                              type: 'line',
                              x0: 1.0,
                              x1: 1.0,
                              y0: 0,
                              y1: 1,
                              yref: 'paper',
                              line: { color: gridColor, width: 1, dash: 'dash' },
                            }],
                          }}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: '100%' }}
                        />
                      </div>
                    );
                  })()}
                </div>
                )}
              </div>
            )}

            {/* Multi-Path Mode View */}
            {!fullVisualization && backtestResult && (
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                <div style={{ width: '600px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {/* Summary Statistics */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '0.75rem',
                    padding: '1.5rem',
                    backgroundColor: 'var(--card-bg)',
                    borderRadius: '0.5rem',
                    border: '1px solid var(--border-color)',
                  }}>
                    <h4 style={{ margin: 0, fontSize: '1rem', marginBottom: '0.5rem', gridColumn: '1 / -1' }}>
                      Summary Statistics ({backtestResult.summary_stats.num_rebalances} rebalances)
                    </h4>
                    {[
                      { label: 'Mean P&L', value: `$${backtestResult.summary_stats.mean_pnl.toFixed(2)}` },
                      { label: 'Std Dev', value: `$${backtestResult.summary_stats.std_pnl.toFixed(2)}` },
                      { label: 'Sharpe Ratio', value: backtestResult.summary_stats.sharpe_ratio.toFixed(3) },
                      { label: 'Median P&L', value: `$${backtestResult.summary_stats.median_pnl.toFixed(2)}` },
                      { label: 'Min P&L', value: `$${backtestResult.summary_stats.min_pnl.toFixed(2)}` },
                      { label: 'Max P&L', value: `$${backtestResult.summary_stats.max_pnl.toFixed(2)}` },
                      { label: 'VaR (95%)', value: `$${backtestResult.summary_stats.var_95.toFixed(2)}` },
                      { label: 'CVaR (95%)', value: `$${backtestResult.summary_stats.cvar_95.toFixed(2)}` },
                    ].map((stat, idx) => (
                      <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', padding: '0.5rem', backgroundColor: 'var(--bg-secondary)', borderRadius: '0.25rem' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>{stat.label}:</span>
                        <span style={{ fontWeight: '500' }}>{stat.value}</span>
                      </div>
                    ))}
                  </div>

                  {/* P&L Distribution Histogram */}
                  <div className="chart-container">
                    <Plot
                      key="pnl-distribution-main"
                      data={[
                        {
                          x: backtestResult.final_pnl_distribution,
                          type: 'histogram',
                          marker: {
                            color: '#4CAF50',
                            line: { color: '#2E7D32', width: 1 },
                          },
                          nbinsx: 40,
                        },
                      ]}
                      layout={{
                        title: { text: `Final P&L Distribution (${backtestResult.final_pnl_distribution.length} paths)`, font: { size: 14 } },
                        xaxis: { title: axisTitle('P&L ($)'), gridcolor: gridColor },
                        yaxis: { title: axisTitle('Frequency'), gridcolor: gridColor },
                        showlegend: false,
                        paper_bgcolor: plotBgColor,
                        plot_bgcolor: plotBgColor,
                        font: { color: plotTextColor },
                        height: 400,
                        margin: { l: 60, r: 40, t: 50, b: 60 },
                        shapes: [
                          {
                            type: 'line',
                            x0: backtestResult.summary_stats.mean_pnl,
                            x1: backtestResult.summary_stats.mean_pnl,
                            y0: 0,
                            y1: 1,
                            yref: 'paper',
                            line: { color: '#FF5722', width: 2, dash: 'dash' },
                          },
                          {
                            type: 'line',
                            x0: backtestResult.summary_stats.var_95,
                            x1: backtestResult.summary_stats.var_95,
                            y0: 0,
                            y1: 1,
                            yref: 'paper',
                            line: { color: '#F44336', width: 2, dash: 'dot' },
                          },
                        ],
                        annotations: [
                          {
                            x: backtestResult.summary_stats.mean_pnl,
                            y: 1.05,
                            yref: 'paper',
                            text: 'Mean',
                            showarrow: false,
                            font: { size: 10, color: '#FF5722' },
                          },
                          {
                            x: backtestResult.summary_stats.var_95,
                            y: 1.05,
                            yref: 'paper',
                            text: 'VaR 95%',
                            showarrow: false,
                            font: { size: 10, color: '#F44336' },
                          },
                        ],
                      }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%' }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default BacktestingPage;
