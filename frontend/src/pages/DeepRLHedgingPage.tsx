import React, { useState } from 'react';
import './DeepRLHedgingPage.css';
import Plot from 'react-plotly.js';
import { rlHedgingApi, RLInferenceResponse } from '../services/rlHedgingApi';
import { useTheme } from '../contexts/ThemeContext';

type AgentType = 'ppo' | 'sac' | 'td3' | 'deep_hedging' | 'ais_hedging';

interface AgentConfig {
  type: AgentType;
  name: string;
  description: string;
  modelFile: File | null;
  isDemo: boolean;
}

interface SimulationConfig {
  s0: number;
  strike: number;
  maturity: number;
  volatility: number;
  riskFreeRate: number;
  nSteps: number;
  transactionCostBps: number;
}

const DeepRLHedgingPage: React.FC = () => {
  const { theme } = useTheme();

  const plotBgColor = theme === 'dark' ? '#0b1220' : '#ffffff';
  const plotTextColor = theme === 'dark' ? '#e5e7eb' : '#0f172a';
  const gridColor = theme === 'dark' ? '#334155' : '#e2e8f0';
  const axisTitle = (text: string) => ({ text, font: { color: plotTextColor } });

  const [selectedAgent, setSelectedAgent] = useState<AgentType | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);

  const [simulationConfig, setSimulationConfig] = useState<SimulationConfig>({
    s0: 100,
    strike: 100,
    maturity: 30,
    volatility: 0.25,
    riskFreeRate: 0.05,
    nSteps: 30,
    transactionCostBps: 5.0,
  });

  const demoAgents = [
    {
      type: 'ppo' as AgentType,
      name: 'PPO Agent',
      description: 'Proximal Policy Optimization - stable on-policy RL',
      icon: '🎯',
      features: [
        'Clipped objective for stable training',
        'Handles continuous action spaces',
        'Good for high-dimensional state spaces',
      ],
    },
    {
      type: 'sac' as AgentType,
      name: 'SAC Agent',
      description: 'Soft Actor-Critic - maximum entropy off-policy RL',
      icon: '🔄',
      features: [
        'Automatic temperature tuning',
        'Sample-efficient off-policy learning',
        'Robust to hyperparameter choices',
      ],
    },
    {
      type: 'td3' as AgentType,
      name: 'TD3 Agent',
      description: 'Twin Delayed DDPG - robust deterministic policy',
      icon: '⚡',
      features: [
        'Reduced overestimation bias',
        'Delayed policy updates',
        'Target policy smoothing',
      ],
    },
    {
      type: 'deep_hedging' as AgentType,
      name: 'Deep Hedging Network',
      description: 'End-to-end neural network for hedging optimization',
      icon: '🧠',
      features: [
        'Direct optimization of hedging P&L',
        'Handles transaction costs natively',
        'No-arbitrage constraints via convex layers',
      ],
    },
    {
      type: 'ais_hedging' as AgentType,
      name: 'AIS Hedging Agent',
      description: 'Adaptive Importance Sampling for tail risk hedging',
      icon: '🎲',
      features: [
        'Efficient rare event simulation',
        'CVaR-focused hedging strategies',
        'Cross-entropy method optimization',
      ],
    },
  ];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setModelFile(file);
      setError(null);
    }
  };

  const handleRunDemo = async (agentType: AgentType) => {
    setIsRunning(true);
    setError(null);
    setSelectedAgent(agentType);

    try {
      const result = await rlHedgingApi.runInference({
        agent_type: agentType,
        environment_config: {
          s0: simulationConfig.s0,
          strike: simulationConfig.strike,
          maturity: simulationConfig.maturity,
          volatility: simulationConfig.volatility,
          risk_free_rate: simulationConfig.riskFreeRate,
          n_steps: simulationConfig.nSteps,
          transaction_cost_bps: simulationConfig.transactionCostBps,
        },
        use_demo_model: true,
      });

      setResults({
        timeGrid: result.time_grid,
        spotPath: result.spot_path,
        hedgePositions: result.hedge_positions,
        pnl: result.pnl,
        finalPnL: result.final_pnl,
        sharpeRatio: result.sharpe_ratio,
        maxDrawdown: result.max_drawdown,
        numRebalances: result.num_rebalances,
      });
    } catch (err: any) {
      setError(err.message || 'Simulation failed');
    } finally {
      setIsRunning(false);
    }
  };

  const handleRunCustomModel = async () => {
    if (!modelFile) {
      setError('Please upload a model file first');
      return;
    }

    setIsRunning(true);
    setError(null);

    try {
      // TODO: Implement actual model upload and inference API call
      setError('Custom model inference coming soon - use demo agents for now');
    } catch (err: any) {
      setError(err.message || 'Model inference failed');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="deep-rl-hedging-page">
      <div className="page-header">
        <h1>Deep RL Hedging</h1>
        <p className="page-subtitle">
          Demonstrate pre-trained reinforcement learning and deep hedging agents on option hedging tasks
        </p>
      </div>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
          <button className="error-close" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <div className="rl-hedging-layout">
        {/* Agent Selection Panel */}
        <div className="agents-panel">
          <h2>Demo Agents</h2>
          <p className="panel-description">
            Pre-trained agents ready for inference on demonstration scenarios
          </p>

          <div className="agents-grid">
            {demoAgents.map((agent) => (
              <div
                key={agent.type}
                className={`agent-card ${selectedAgent === agent.type ? 'selected' : ''}`}
                onClick={() => !isRunning && handleRunDemo(agent.type)}
              >
                <div className="agent-icon">{agent.icon}</div>
                <h3 className="agent-name">{agent.name}</h3>
                <p className="agent-description">{agent.description}</p>
                <ul className="agent-features">
                  {agent.features.map((feature, idx) => (
                    <li key={idx}>{feature}</li>
                  ))}
                </ul>
                <button
                  className="run-demo-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRunDemo(agent.type);
                  }}
                  disabled={isRunning}
                >
                  {isRunning && selectedAgent === agent.type ? 'Running...' : 'Run Demo'}
                </button>
              </div>
            ))}
          </div>

          {/* Custom Model Upload */}
          <div className="custom-model-section">
            <h3>Upload Custom Model</h3>
            <div className="upload-area">
              <input
                type="file"
                accept=".pt,.pth,.h5,.pkl,.onnx"
                onChange={handleFileUpload}
                id="model-upload"
                style={{ display: 'none' }}
              />
              <label htmlFor="model-upload" className="upload-label">
                <span className="upload-icon">📤</span>
                <span className="upload-text">
                  {modelFile ? modelFile.name : 'Choose model file (.pt, .h5, .pkl, .onnx)'}
                </span>
              </label>
              <button
                className="btn btn-primary"
                onClick={handleRunCustomModel}
                disabled={!modelFile || isRunning}
              >
                Run Custom Model
              </button>
            </div>
          </div>
        </div>

        {/* Simulation Configuration */}
        <div className="config-panel">
          <h2>Environment Configuration</h2>
          <div className="config-grid">
            <div className="config-field">
              <label>Initial Spot (S₀)</label>
              <input
                type="number"
                value={simulationConfig.s0}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, s0: parseFloat(e.target.value) })}
              />
            </div>
            <div className="config-field">
              <label>Strike (K)</label>
              <input
                type="number"
                value={simulationConfig.strike}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, strike: parseFloat(e.target.value) })}
              />
            </div>
            <div className="config-field">
              <label>Maturity (days)</label>
              <input
                type="number"
                value={simulationConfig.maturity}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, maturity: parseInt(e.target.value) })}
              />
            </div>
            <div className="config-field">
              <label>Volatility (σ)</label>
              <input
                type="number"
                step="0.01"
                value={simulationConfig.volatility}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, volatility: parseFloat(e.target.value) })}
              />
            </div>
            <div className="config-field">
              <label>Risk-Free Rate (r)</label>
              <input
                type="number"
                step="0.01"
                value={simulationConfig.riskFreeRate}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, riskFreeRate: parseFloat(e.target.value) })}
              />
            </div>
            <div className="config-field">
              <label>Steps</label>
              <input
                type="number"
                value={simulationConfig.nSteps}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, nSteps: parseInt(e.target.value) })}
              />
            </div>
            <div className="config-field">
              <label>Transaction Cost (bps)</label>
              <input
                type="number"
                step="0.1"
                value={simulationConfig.transactionCostBps}
                onChange={(e) => setSimulationConfig({ ...simulationConfig, transactionCostBps: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        </div>

        {/* Results Visualization */}
        {results && (
          <div className="results-panel">
            <h2>Agent Performance</h2>

            {/* Summary Stats */}
            <div className="summary-stats">
              <div className="stat-card">
                <div className="stat-label">Final P&L</div>
                <div className="stat-value">${results.finalPnL.toFixed(2)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Sharpe Ratio</div>
                <div className="stat-value">{results.sharpeRatio.toFixed(3)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Max Drawdown</div>
                <div className="stat-value">${results.maxDrawdown.toFixed(2)}</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Rebalances</div>
                <div className="stat-value">{results.numRebalances}</div>
              </div>
            </div>

            {/* Charts */}
            <div className="charts-grid">
              <div className="chart-container">
                <Plot
                  data={[
                    {
                      x: results.timeGrid,
                      y: results.spotPath,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Spot Price',
                      line: { color: '#2E86AB', width: 2 },
                    },
                    {
                      x: [0, results.timeGrid.length - 1],
                      y: [simulationConfig.strike, simulationConfig.strike],
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Strike',
                      line: { color: '#f44336', width: 2, dash: 'dash' },
                    },
                  ]}
                  layout={{
                    title: { text: 'Price Dynamics' },
                    xaxis: { title: axisTitle('Time Step'), gridcolor: gridColor },
                    yaxis: { title: axisTitle('Spot Price ($)'), gridcolor: gridColor },
                    paper_bgcolor: plotBgColor,
                    plot_bgcolor: plotBgColor,
                    font: { color: plotTextColor },
                    height: 300,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>

              <div className="chart-container">
                <Plot
                  data={[
                    {
                      x: results.timeGrid,
                      y: results.pnl,
                      type: 'scatter',
                      mode: 'lines',
                      fill: 'tozeroy',
                      name: 'P&L',
                      line: { color: results.finalPnL >= 0 ? '#4CAF50' : '#f44336', width: 2 },
                    },
                  ]}
                  layout={{
                    title: { text: 'Portfolio P&L' },
                    xaxis: { title: axisTitle('Time Step'), gridcolor: gridColor },
                    yaxis: { title: axisTitle('P&L ($)'), gridcolor: gridColor },
                    showlegend: false,
                    paper_bgcolor: plotBgColor,
                    plot_bgcolor: plotBgColor,
                    font: { color: plotTextColor },
                    height: 300,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>

              <div className="chart-container">
                <Plot
                  data={[
                    {
                      x: results.timeGrid,
                      y: results.hedgePositions,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Hedge Position',
                      line: { color: '#9C27B0', width: 2 },
                    },
                  ]}
                  layout={{
                    title: { text: 'Hedge Position Over Time' },
                    xaxis: { title: axisTitle('Time Step'), gridcolor: gridColor },
                    yaxis: { title: axisTitle('Position (shares)'), gridcolor: gridColor },
                    showlegend: false,
                    paper_bgcolor: plotBgColor,
                    plot_bgcolor: plotBgColor,
                    font: { color: plotTextColor },
                    height: 300,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>

              <div className="chart-container">
                <Plot
                  data={[
                    {
                      x: results.timeGrid.slice(1),
                      y: results.hedgePositions.slice(1).map((h: number, i: number) =>
                        Math.abs(h - results.hedgePositions[i])
                      ),
                      type: 'bar',
                      name: 'Rebalancing',
                      marker: { color: '#FF9800' },
                    },
                  ]}
                  layout={{
                    title: { text: 'Rebalancing Activity' },
                    xaxis: { title: axisTitle('Time Step'), gridcolor: gridColor },
                    yaxis: { title: axisTitle('Position Change'), gridcolor: gridColor },
                    showlegend: false,
                    paper_bgcolor: plotBgColor,
                    plot_bgcolor: plotBgColor,
                    font: { color: plotTextColor },
                    height: 300,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DeepRLHedgingPage;
