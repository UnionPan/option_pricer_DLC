"""Backtesting schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class LiabilitySpec(BaseModel):
    """Specification of option liability to hedge."""
    option_type: str = Field(..., description="'call' or 'put'")
    strike: float = Field(..., description="Strike price")
    maturity_days: int = Field(..., description="Days to maturity")
    quantity: float = Field(default=-1.0, description="Quantity (negative = short position)")


class HedgeOptionSpec(BaseModel):
    """Specification of hedge option instrument."""
    option_type: str = Field(..., description="'call' or 'put'")
    strike: float = Field(..., description="Strike price")
    maturity_days: int = Field(..., description="Days to maturity")


class BacktestRequest(BaseModel):
    """Request to run backtesting simulation."""
    model: str = Field(..., description="Model name: 'gbm', 'heston', 'ou'")
    parameters: Dict[str, float] = Field(..., description="Calibrated model parameters")
    liability_spec: LiabilitySpec = Field(..., description="Option liability specification")
    hedging_strategy: str = Field(
        default="delta_hedge",
        description="Hedging strategy: 'delta_hedge', 'delta_gamma_hedge', 'delta_vega_hedge', 'delta_gamma_vega_hedge', 'no_hedge'"
    )
    hedge_options: Optional[List[HedgeOptionSpec]] = Field(
        default=None,
        description="Optional hedge option instruments for gamma/vega hedging."
    )
    heston_pricer: Optional[str] = Field(
        default="mgf",
        description="Heston pricing method: 'mgf' (fast) or 'analytical' (slow)."
    )
    s0: float = Field(..., description="Initial spot price")
    n_steps: int = Field(default=252, description="Number of time steps")
    n_paths: int = Field(default=1000, description="Number of Monte Carlo paths")
    dt: float = Field(default=1/252, description="Time step size (years)")
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate (annualized)")
    transaction_cost_bps: float = Field(default=5.0, description="Transaction cost in basis points")
    rebalance_threshold: float = Field(default=0.05, description="Min delta change to rebalance")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    full_visualization: bool = Field(
        default=True,
        description="Generate full visualization data (option chains, IV surface). Set to False for faster computation."
    )


class Greeks(BaseModel):
    """Greeks over time."""
    delta: List[float]
    gamma: List[float]
    vega: List[float]
    theta: List[float]


class Transaction(BaseModel):
    """Single transaction event."""
    time: float
    spot: float
    action: str
    shares_traded: Optional[float] = None
    contracts_traded: Optional[float] = None
    option_strike: Optional[float] = None
    option_type: Optional[str] = None
    hedge_leg: Optional[int] = None
    cost: float
    delta: Optional[float] = None
    transaction_cost: Optional[float] = None


class SummaryStats(BaseModel):
    """Summary statistics of backtest."""
    mean_pnl: float
    std_pnl: float
    median_pnl: float
    min_pnl: float
    max_pnl: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    num_rebalances: int
    total_transaction_costs: float


class OptionChainData(BaseModel):
    """Option chain data at a single timestep."""
    time_step: int
    time: float
    spot: float
    volatility: float
    options: List[Dict[str, Any]]


class IVSurfaceData(BaseModel):
    """IV surface data for 3D visualization."""
    moneyness: List[float]
    ttm: List[float]
    iv: List[float]
    option_type: List[str]
    time_step: List[int]
    spot: List[float]


class BacktestResponse(BaseModel):
    """Response from backtesting simulation."""
    time_grid: List[float]
    representative_path: List[float]
    all_paths: List[List[float]]
    variance_path: Optional[List[float]] = Field(default=None, description="Variance path (Heston only)")
    volatility_path: List[float] = Field(..., description="Volatility path (sqrt of variance)")
    hedge_positions: List[float]
    hedge_option_positions: Optional[List[List[float]]] = None
    hedge_option_value: Optional[List[List[float]]] = None
    cash: List[float]
    portfolio_value: List[float]
    option_value: List[float]
    pnl: List[float]
    greeks: Greeks
    transactions: List[Transaction]
    summary_stats: SummaryStats
    final_pnl_distribution: List[float]
    option_chains: Optional[List[OptionChainData]] = Field(
        default=None,
        description="Option chain at each timestep (only if full_visualization=True)"
    )
    iv_surface: Optional[IVSurfaceData] = Field(
        default=None,
        description="Aggregated IV surface for 3D visualization (only if full_visualization=True)"
    )
    liability_spec: LiabilitySpec
    hedge_option_specs: Optional[List[HedgeOptionSpec]] = None
    hedging_strategy: str
    model: str
    parameters: Dict[str, float]
