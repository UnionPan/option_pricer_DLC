"""Backtesting endpoints for hedging strategies."""

from fastapi import APIRouter, HTTPException
from backend.services.backtesting_service import BacktestingService
from backend.schemas.backtesting import (
    BacktestRequest,
    BacktestResponse,
    Greeks,
    Transaction,
    SummaryStats,
    OptionChainData,
    IVSurfaceData,
)

router = APIRouter()


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run backtesting simulation of hedging strategy.

    This endpoint simulates price dynamics using calibrated P-measure models
    and runs analytical hedging strategies (Delta hedging, Gamma hedging, etc.)
    to evaluate hedging performance.

    Args:
        request: BacktestRequest with model parameters and hedging configuration

    Returns:
        BacktestResponse with detailed results including:
            - Price paths
            - P&L evolution
            - Greeks evolution
            - Hedging transactions
            - Summary statistics
    """
    try:
        result = BacktestingService.run_backtest(
            model=request.model,
            parameters=request.parameters,
            liability_spec=request.liability_spec.dict(),
            hedging_strategy=request.hedging_strategy,
            hedge_options=[opt.dict() for opt in request.hedge_options] if request.hedge_options else None,
            heston_pricer=request.heston_pricer,
            s0=request.s0,
            n_steps=request.n_steps,
            n_paths=request.n_paths,
            dt=request.dt,
            risk_free_rate=request.risk_free_rate,
            transaction_cost_bps=request.transaction_cost_bps,
            rebalance_threshold=request.rebalance_threshold,
            random_seed=request.random_seed,
            full_visualization=request.full_visualization,
        )

        return BacktestResponse(
            time_grid=result['time_grid'],
            representative_path=result['representative_path'],
            all_paths=result['all_paths'],
            variance_path=result.get('variance_path'),
            volatility_path=result['volatility_path'],
            hedge_positions=result['hedge_positions'],
            hedge_option_positions=result.get('hedge_option_positions'),
            hedge_option_value=result.get('hedge_option_value'),
            cash=result['cash'],
            portfolio_value=result['portfolio_value'],
            option_value=result['option_value'],
            pnl=result['pnl'],
            greeks=Greeks(**result['greeks']),
            transactions=[Transaction(**t) for t in result['transactions']],
            summary_stats=SummaryStats(**result['summary_stats']),
            final_pnl_distribution=result['final_pnl_distribution'],
            option_chains=[OptionChainData(**chain) for chain in result['option_chains']] if result['option_chains'] else None,
            iv_surface=IVSurfaceData(**result['iv_surface']) if result['iv_surface'] else None,
            liability_spec=request.liability_spec,
            hedge_option_specs=result.get('hedge_option_specs'),
            hedging_strategy=result['hedging_strategy'],
            model=result['model'],
            parameters=result['parameters'],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
