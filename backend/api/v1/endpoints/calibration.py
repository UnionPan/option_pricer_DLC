"""Calibration endpoints for P-measure and Q-measure models."""
from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime

from backend.services.calibration_service import CalibrationService
from backend.schemas.calibration import (
    # P-measure schemas
    FetchDataRequest,
    OHLCVData,
    CalibrationRequest,
    CalibrationResult,
    CalibrationDiagnostics,
    # Q-measure schemas
    FetchOptionChainRequest,
    OptionChainData,
    QMeasureCalibrationRequest,
    QMeasureCalibrationResult,
    VolSurfaceRequest,
    VolSurfaceResponse,
)
from backend.schemas.simulation import SimulationRequest, SimulationResponse

router = APIRouter()


# ========== P-measure (Physical) Calibration Endpoints ==========

@router.post("/fetch-data", response_model=List[OHLCVData])
async def fetch_ohlcv_data(request: FetchDataRequest):
    """Fetch OHLCV data from yfinance for P-measure calibration."""
    try:
        data = CalibrationService.fetch_ohlcv_data(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate", response_model=List[CalibrationResult])
async def calibrate_pmeasure(request: CalibrationRequest):
    """Calibrate P-measure models from historical data."""
    try:
        # Fetch OHLCV data first
        ohlcv_data = CalibrationService.fetch_ohlcv_data(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        results = []
        for data in ohlcv_data:
            ticker = data['ticker']
            prices = data['close']
            dates = data['dates']

            # Run calibration
            parameters, diagnostics = CalibrationService.calibrate_model(
                ticker=ticker,
                prices=prices,
                dates=dates,
                model=request.model,
                method=request.method,
                include_drift=request.include_drift,
            )

            # Format result
            result = CalibrationResult(
                ticker=ticker,
                model=request.model,
                parameters=parameters,
                diagnostics=CalibrationDiagnostics(**diagnostics),
                timestamp=datetime.now().isoformat(),
                method=request.method,
                measure="P-measure",
            )
            results.append(result)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Q-measure (Risk-Neutral) Calibration Endpoints ==========

@router.post("/fetch-option-chain", response_model=OptionChainData)
async def fetch_option_chain(request: FetchOptionChainRequest):
    """Fetch option chain data from yfinance for Q-measure calibration."""
    try:
        chain_data = CalibrationService.fetch_option_chain(
            ticker=request.ticker,
            reference_date=request.reference_date,
            risk_free_rate=request.risk_free_rate,
            expiry=request.expiry,
        )
        return chain_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate-qmeasure", response_model=QMeasureCalibrationResult)
async def calibrate_qmeasure(request: QMeasureCalibrationRequest):
    """Calibrate Q-measure models from option chain data."""
    try:
        # Run Q-measure calibration
        parameters, diagnostics = CalibrationService.calibrate_qmeasure_model(
            ticker=request.ticker,
            model=request.model,
            reference_date=request.reference_date,
            risk_free_rate=request.risk_free_rate,
            filter_params=request.filter_params,
            calibration_method=request.calibration_method,
            maxiter=request.maxiter,
        )

        # Format result
        result = QMeasureCalibrationResult(
            ticker=request.ticker,
            model=request.model,
            parameters=parameters,
            diagnostics=diagnostics,
            timestamp=datetime.now().isoformat(),
            measure="Q-measure",
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate", response_model=SimulationResponse)
async def simulate_counterfactual(request: SimulationRequest):
    """Simulate counterfactual paths from calibrated parameters."""
    try:
        result = CalibrationService.simulate_model(
            model=request.model,
            parameters=request.parameters,
            s0=request.s0,
            n_steps=request.n_steps,
            n_paths=request.n_paths,
            dt=request.dt,
            max_paths_return=request.max_paths_return,
            random_seed=request.random_seed,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vol-surface", response_model=VolSurfaceResponse)
async def generate_vol_surface(request: VolSurfaceRequest):
    """Generate 3D volatility surface from calibrated Q-measure model."""
    try:
        surface_data = CalibrationService.generate_vol_surface(
            ticker=request.ticker,
            model=request.model,
            parameters=request.parameters,
            spot_price=request.spot_price,
            risk_free_rate=request.risk_free_rate,
            dividend_yield=request.dividend_yield,
            n_strikes=request.n_strikes,
            n_maturities=request.n_maturities,
            strike_range=tuple(request.strike_range),
            maturity_range=tuple(request.maturity_range),
        )

        return VolSurfaceResponse(
            strikes=surface_data['strikes'],
            maturities=surface_data['maturities'],
            vols=surface_data['vols'],
            surface_type=surface_data['surface_type'],
            ticker=request.ticker,
            model=request.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
