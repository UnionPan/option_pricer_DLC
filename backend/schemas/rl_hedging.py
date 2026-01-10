"""
Schemas for Deep RL Hedging API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal


class EnvironmentConfig(BaseModel):
    """Configuration for the hedging environment."""
    s0: float = Field(default=100.0, description="Initial spot price")
    strike: float = Field(default=100.0, description="Strike price")
    maturity: int = Field(default=30, description="Maturity in days")
    volatility: float = Field(default=0.25, description="Volatility")
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate")
    n_steps: int = Field(default=30, description="Number of time steps")
    transaction_cost_bps: float = Field(default=5.0, description="Transaction cost in basis points")


class RLInferenceRequest(BaseModel):
    """Request to run RL agent inference on a demonstration environment."""
    agent_type: Literal['ppo', 'sac', 'td3', 'deep_hedging', 'ais_hedging']
    environment_config: EnvironmentConfig
    use_demo_model: bool = Field(default=True, description="Use pre-trained demo model")
    model_path: Optional[str] = Field(default=None, description="Path to custom model file")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class RLInferenceResponse(BaseModel):
    """Response from RL agent inference."""
    agent_type: str
    time_grid: List[int]
    spot_path: List[float]
    hedge_positions: List[float]
    pnl: List[float]
    final_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    num_rebalances: int
    total_transaction_costs: float
    environment_config: EnvironmentConfig
