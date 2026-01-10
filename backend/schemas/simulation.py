"""
Schemas for counterfactual simulation endpoints.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class SimulationRequest(BaseModel):
    """Request to simulate paths from calibrated parameters."""
    model: str = Field(..., description="Model name (gbm, ou, heston, rough_bergomi, merton_jump, garch, regime_switching_gbm)")
    parameters: Dict[str, float] = Field(..., description="Calibrated model parameters")
    s0: float = Field(..., description="Initial spot price")
    n_steps: int = Field(default=252, description="Number of time steps")
    n_paths: int = Field(default=200, description="Number of simulated paths")
    dt: float = Field(default=1.0 / 252.0, description="Time step size in years")
    max_paths_return: int = Field(default=20, description="Max sample paths to return")
    random_seed: Optional[int] = Field(default=None, description="Optional RNG seed")


class SimulationStats(BaseModel):
    """Summary statistics for final spot distribution."""
    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    min: float
    max: float


class SimulationResponse(BaseModel):
    """Simulation results and summary stats."""
    model: str
    n_steps: int
    n_paths: int
    dt: float
    time_grid: List[float]
    mean_path: List[float]
    sample_paths: List[List[float]]
    stats: SimulationStats
