import React, { useState } from 'react';
import './CalibrationPage.css';
import AssetSelector from '../components/calibration/AssetSelector';
import ModelConfiguration from '../components/calibration/ModelConfiguration';
import OHLCVChart from '../components/calibration/OHLCVChart';
import CalibrationResults from '../components/calibration/CalibrationResults';
import SimulationResults from '../components/calibration/SimulationResults';
import VolatilitySurface3D from '../components/calibration/VolatilitySurface3D';
import { calibrationApi, OptionChainData, QMeasureCalibrationResponse, SimulationResponse, VolSurfaceResponse } from '../services/calibrationApi';

export interface Asset {
  ticker: string;
  name: string;
}

export interface CalibrationConfig {
  model: string;
  startDate: string;
  endDate: string;
  calibrationMethod: string;
  includeDrift: boolean;
  riskFreeRate?: number;
}

export interface CalibrationResult {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  diagnostics: {
    logLikelihood?: number;
    aic?: number;
    bic?: number;
    mean_ess?: number;
    n_particles?: number;
    variogram_r2?: number;
    errorMetrics?: Record<string, number>;
    note?: string;
  };
  timestamp: string;
  method?: string;
  measure?: string;
}

type MeasureType = 'P' | 'Q';

const CalibrationPage: React.FC = () => {
  const [measureType, setMeasureType] = useState<MeasureType>('P');
  const [selectedAssets, setSelectedAssets] = useState<Asset[]>([]);
  const [config, setConfig] = useState<CalibrationConfig>({
    model: 'heston',
    startDate: '',
    endDate: '',
    calibrationMethod: 'mle',
    includeDrift: true,
    riskFreeRate: 0.05,
  });
  const [ohlcvData, setOhlcvData] = useState<any[]>([]);
  const [optionChainData, setOptionChainData] = useState<OptionChainData | null>(null);
  const [calibrationResults, setCalibrationResults] = useState<CalibrationResult[]>([]);
  const [qMeasureResult, setQMeasureResult] = useState<QMeasureCalibrationResponse | null>(null);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResponse | null>(null);
  const [simulationTicker, setSimulationTicker] = useState<string>('');
  const [simulationConfig, setSimulationConfig] = useState({
    nPaths: 200,
    nSteps: 252,
    dt: 1 / 252,
    s0: 100,
  });
  const [volSurfaceData, setVolSurfaceData] = useState<VolSurfaceResponse | null>(null);
  const [isGeneratingSurface, setIsGeneratingSurface] = useState(false);
  const [surfaceError, setSurfaceError] = useState<string | null>(null);

  React.useEffect(() => {
    if (measureType !== 'P' || !simulationTicker) {
      return;
    }
    const match = ohlcvData.find((d) => d.ticker === simulationTicker);
    if (match && match.close.length > 0) {
      setSimulationConfig((prev) => ({
        ...prev,
        s0: match.close[match.close.length - 1],
      }));
    }
  }, [measureType, simulationTicker, ohlcvData]);

  const handleAssetChange = (assets: Asset[]) => {
    setSelectedAssets(assets);
    // Clear previous data when assets change
    setOhlcvData([]);
    setOptionChainData(null);
    setCalibrationResults([]);
    setQMeasureResult(null);
    setSimulationResult(null);
    setSimulationError(null);
    setVolSurfaceData(null);
    setSurfaceError(null);
    setError(null);
  };

  const handleMeasureTypeChange = (type: MeasureType) => {
    setMeasureType(type);
    // Clear data when switching measure types
    setOhlcvData([]);
    setOptionChainData(null);
    setCalibrationResults([]);
    setQMeasureResult(null);
    setSimulationResult(null);
    setSimulationError(null);
    setVolSurfaceData(null);
    setSurfaceError(null);
    setError(null);

    // Set appropriate defaults for Q-measure
    if (type === 'Q') {
      setConfig({
        ...config,
        model: 'heston',
        calibrationMethod: 'differential_evolution',
        riskFreeRate: 0.05,
      });
    }
  };

  const handleConfigChange = (newConfig: Partial<CalibrationConfig>) => {
    setConfig({ ...config, ...newConfig });
  };

  const handleFetchData = async () => {
    if (selectedAssets.length === 0) {
      setError('Please select at least one asset');
      return;
    }

    setIsLoadingData(true);
    setError(null);

    try {
      if (measureType === 'P') {
        // Fetch OHLCV data for P-measure
        if (!config.startDate || !config.endDate) {
          setError('Please select start and end dates');
          setIsLoadingData(false);
          return;
        }

        const tickers = selectedAssets.map(a => a.ticker);
        const data = await calibrationApi.fetchOHLCV(
          tickers,
          config.startDate,
          config.endDate
        );
        setOhlcvData(data);
      } else {
        // Fetch option chain for Q-measure (single asset only)
        if (selectedAssets.length > 1) {
          setError('Q-measure calibration supports single asset only');
          setIsLoadingData(false);
          return;
        }

        const ticker = selectedAssets[0].ticker;
        const chainData = await calibrationApi.fetchOptionChain(
          ticker,
          undefined, // Use today's date
          config.riskFreeRate || 0.05
        );
        setOptionChainData(chainData);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch data');
    } finally {
      setIsLoadingData(false);
    }
  };

  const handleCalibrate = async () => {
    setIsCalibrating(true);
    setError(null);

    try {
      if (measureType === 'P') {
        if (ohlcvData.length === 0) {
          setError('Please fetch data first');
          setIsCalibrating(false);
          return;
        }

        const tickers = selectedAssets.map(a => a.ticker);
      const results = await calibrationApi.calibrate({
        tickers,
        model: config.model,
        startDate: config.startDate,
        endDate: config.endDate,
        method: config.calibrationMethod,
        includeDrift: config.includeDrift,
      });
      setCalibrationResults(results);
        if (results.length > 0) {
          setSimulationTicker(results[0].ticker);
          const match = ohlcvData.find((d) => d.ticker === results[0].ticker);
          if (match && match.close.length > 0) {
            setSimulationConfig((prev) => ({
              ...prev,
              s0: match.close[match.close.length - 1],
            }));
          }
        }
      } else {
        if (!optionChainData) {
          setError('Please fetch option chain first');
          setIsCalibrating(false);
          return;
        }

        const result = await calibrationApi.calibrateQMeasure({
          ticker: selectedAssets[0].ticker,
          model: config.model,
          riskFreeRate: config.riskFreeRate || 0.05,
          calibrationMethod: config.calibrationMethod === 'mle' ? 'differential_evolution' : config.calibrationMethod,
          maxiter: 1000,
          filterParams: {
            min_volume: 10,
            min_open_interest: 50,
            max_spread_pct: 0.5,
            moneyness_range: [0.8, 1.2],
          },
        });
        setQMeasureResult(result);
        if (optionChainData) {
          setSimulationConfig((prev) => ({
            ...prev,
            s0: optionChainData.spot_price,
          }));
        }
      }
    } catch (err: any) {
      setError(err.message || 'Calibration failed');
    } finally {
      setIsCalibrating(false);
    }
  };

  const handleDownloadResults = () => {
    const dataToDownload = measureType === 'P' ? calibrationResults : qMeasureResult;
    if (!dataToDownload || (Array.isArray(dataToDownload) && dataToDownload.length === 0)) return;

    const dataStr = JSON.stringify(dataToDownload, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

    const measureLabel = measureType === 'P' ? 'pmeasure' : 'qmeasure';
    const exportFileDefaultName = `calibration_${measureLabel}_${new Date().toISOString().split('T')[0]}.json`;

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const hasResults = measureType === 'P' ? calibrationResults.length > 0 : qMeasureResult !== null;
  const hasData = measureType === 'P' ? ohlcvData.length > 0 : optionChainData !== null;

  const selectedResult = measureType === 'P'
    ? calibrationResults.find((r) => r.ticker === simulationTicker) || calibrationResults[0]
    : qMeasureResult;

  const handleRunSimulation = async () => {
    if (!selectedResult) {
      setSimulationError('No calibration results available');
      return;
    }

    setIsSimulating(true);
    setSimulationError(null);

    try {
      const result = await calibrationApi.simulate({
        model: selectedResult.model,
        parameters: selectedResult.parameters,
        s0: simulationConfig.s0,
        nSteps: simulationConfig.nSteps,
        nPaths: simulationConfig.nPaths,
        dt: simulationConfig.dt,
        maxPathsReturn: 20,
      });
      setSimulationResult(result);
    } catch (err: any) {
      setSimulationError(err.message || 'Simulation failed');
    } finally {
      setIsSimulating(false);
    }
  };

  const handleGenerateVolSurface = async () => {
    if (!qMeasureResult || !optionChainData) {
      setSurfaceError('Q-measure calibration required');
      return;
    }

    setIsGeneratingSurface(true);
    setSurfaceError(null);

    try {
      const surfaceData = await calibrationApi.generateVolSurface({
        ticker: qMeasureResult.ticker,
        model: qMeasureResult.model,
        parameters: qMeasureResult.parameters,
        spotPrice: optionChainData.spot_price,
        riskFreeRate: config.riskFreeRate || 0.05,
        dividendYield: optionChainData.dividend_yield || 0.0,
        nStrikes: 30,
        nMaturities: 20,
        strikeRange: [0.7, 1.3],
        maturityRange: [0.05, 2.0],
      });
      setVolSurfaceData(surfaceData);
    } catch (err: any) {
      setSurfaceError(err.message || 'Volatility surface generation failed');
    } finally {
      setIsGeneratingSurface(false);
    }
  };

  return (
    <div className="calibration-page">
      <div className="page-header">
        <h1>Model Calibration</h1>
        <p className="page-subtitle">
          Calibrate {measureType}-measure models to market data
        </p>
      </div>

      {/* Measure Type Selector */}
      <div className="measure-selector">
        <button
          className={`measure-btn ${measureType === 'P' ? 'active' : ''}`}
          onClick={() => handleMeasureTypeChange('P')}
        >
          <strong>P-measure</strong>
          <span>Historical Data (Real-world)</span>
        </button>
        <button
          className={`measure-btn ${measureType === 'Q' ? 'active' : ''}`}
          onClick={() => handleMeasureTypeChange('Q')}
        >
          <strong>Q-measure</strong>
          <span>Option Prices (Risk-neutral)</span>
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
          <button className="error-close" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="calibration-layout">
        <div className="configuration-card">
          <div className="configuration-top">
            <div className="configuration-left">
              <AssetSelector
                selectedAssets={selectedAssets}
                onAssetChange={handleAssetChange}
              />
            </div>

            <div className="configuration-right">
              <ModelConfiguration
                config={config}
                onConfigChange={handleConfigChange}
                measureType={measureType}
                showDateRange={false}
              />

              {measureType === 'P' && (
                <div className="date-range-panel">
                  <div className="date-range-row">
                    <div className="config-group">
                      <label htmlFor="start-date">Start Date:</label>
                      <input
                        id="start-date"
                        type="date"
                        value={config.startDate}
                        onChange={(e) => handleConfigChange({ startDate: e.target.value })}
                        className="config-input"
                      />
                    </div>

                    <div className="config-group">
                      <label htmlFor="end-date">End Date:</label>
                      <input
                        id="end-date"
                        type="date"
                        value={config.endDate}
                        onChange={(e) => handleConfigChange({ endDate: e.target.value })}
                        className="config-input"
                      />
                    </div>
                  </div>

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
                            handleConfigChange({
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
                </div>
              )}
            </div>
          </div>

          <div className="action-buttons">
            <button
              className="btn btn-primary"
              onClick={handleFetchData}
              disabled={isLoadingData || selectedAssets.length === 0}
            >
              {isLoadingData ? (
                <>
                  <span className="spinner"></span>
                  Fetching Data...
                </>
              ) : (
                measureType === 'P' ? 'Fetch Historical Data' : 'Fetch Option Chain'
              )}
            </button>

            <button
              className="btn btn-secondary"
              onClick={handleCalibrate}
              disabled={isCalibrating || !hasData}
            >
              {isCalibrating ? (
                <>
                  <span className="spinner"></span>
                  Calibrating...
                </>
              ) : (
                'Calibrate Model'
              )}
            </button>
          </div>
        </div>

        <div className="results-panel">
          {measureType === 'P' && ohlcvData.length > 0 && (
            <div className="chart-section">
              <h2>Historical OHLCV Data</h2>
              <OHLCVChart data={ohlcvData} assets={selectedAssets} />
            </div>
          )}

          {measureType === 'Q' && optionChainData && (
            <div className="chart-section">
              <h2>Option Chain Data</h2>
              <div className="option-chain-summary">
                <div className="summary-item">
                  <span className="label">Spot Price:</span>
                  <span className="value">${optionChainData.spot_price.toFixed(2)}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Total Options:</span>
                  <span className="value">{optionChainData.n_options}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Expiries:</span>
                  <span className="value">{optionChainData.expiries.length}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Risk-free Rate:</span>
                  <span className="value">{(optionChainData.risk_free_rate * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          )}

          {hasResults && (
            <div className="results-section">
              <div className="results-header">
                <h2>Calibration Results</h2>
                <button
                  className="btn btn-download"
                  onClick={handleDownloadResults}
                >
                  📥 Download JSON
                </button>
              </div>
              {measureType === 'P' && <CalibrationResults results={calibrationResults} />}
              {measureType === 'Q' && qMeasureResult && (
                <CalibrationResults results={[{
                  model: qMeasureResult.model,
                  parameters: qMeasureResult.parameters,
                  diagnostics: qMeasureResult.diagnostics as any,
                  timestamp: qMeasureResult.timestamp,
                  ticker: qMeasureResult.ticker,
                  method: config.calibrationMethod,
                  measure: 'Q-measure',
                }]} />
              )}
            </div>
          )}

          {measureType === 'Q' && qMeasureResult && optionChainData && (
            <div className="vol-surface-section">
              <div className="vol-surface-header">
                <h2>Volatility Surface</h2>
                <button
                  className="btn btn-primary"
                  onClick={handleGenerateVolSurface}
                  disabled={isGeneratingSurface}
                >
                  {isGeneratingSurface ? (
                    <>
                      <span className="spinner"></span>
                      Generating Surface...
                    </>
                  ) : (
                    'Generate 3D Surface'
                  )}
                </button>
              </div>

              {surfaceError && (
                <div className="error-banner">
                  <span className="error-icon">⚠️</span>
                  <span>{surfaceError}</span>
                  <button className="error-close" onClick={() => setSurfaceError(null)}>×</button>
                </div>
              )}

              <VolatilitySurface3D
                surfaceData={volSurfaceData}
                isLoading={isGeneratingSurface}
                optionChainData={optionChainData}
                spotPrice={optionChainData?.spot_price}
              />
            </div>
          )}

          {measureType === 'P' && hasResults && selectedResult && (
            <div className="simulation-section">
              <div className="simulation-header">
                <h2>Counterfactual Simulation</h2>
              </div>

              <div className="simulation-controls">
                {calibrationResults.length > 1 && (
                  <div className="control-field">
                    <label>Ticker</label>
                    <select
                      value={simulationTicker}
                      onChange={(e) => setSimulationTicker(e.target.value)}
                    >
                      {calibrationResults.map((res) => (
                        <option key={res.ticker} value={res.ticker}>
                          {res.ticker}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <div className="control-field">
                  <label>Initial Spot (S0)</label>
                  <input
                    type="number"
                    value={simulationConfig.s0}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      s0: Number(e.target.value),
                    })}
                  />
                </div>

                <div className="control-field">
                  <label>Paths</label>
                  <input
                    type="number"
                    min={10}
                    value={simulationConfig.nPaths}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      nPaths: Number(e.target.value),
                    })}
                  />
                </div>

                <div className="control-field">
                  <label>Steps</label>
                  <input
                    type="number"
                    min={10}
                    value={simulationConfig.nSteps}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      nSteps: Number(e.target.value),
                    })}
                  />
                </div>

                <div className="control-field">
                  <label>dt (years)</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={simulationConfig.dt}
                    onChange={(e) => setSimulationConfig({
                      ...simulationConfig,
                      dt: Number(e.target.value),
                    })}
                  />
                </div>

                <button
                  className="btn btn-primary"
                  onClick={handleRunSimulation}
                  disabled={isSimulating}
                >
                  {isSimulating ? 'Simulating...' : 'Run Simulation'}
                </button>
              </div>

              {simulationError && (
                <div className="error-banner">
                  <span className="error-icon">⚠️</span>
                  <span>{simulationError}</span>
                  <button className="error-close" onClick={() => setSimulationError(null)}>×</button>
                </div>
              )}

              {simulationResult && (
                <SimulationResults data={simulationResult} />
              )}
            </div>
          )}

          {!hasData && !hasResults && (
            <div className="empty-state">
              <div className="empty-icon">📈</div>
              <h3>No Data Yet</h3>
              <p>
                {measureType === 'P'
                  ? 'Select assets and fetch historical data to begin calibration'
                  : 'Select an asset and fetch option chain to begin calibration'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CalibrationPage;
