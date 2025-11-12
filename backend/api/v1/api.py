"""API router aggregator."""
from fastapi import APIRouter
from backend.api.v1.endpoints import pricing, data, surface, market, smile

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(pricing.router, prefix="/pricing", tags=["Pricing"])
api_router.include_router(data.router, prefix="/data", tags=["Data"])
api_router.include_router(surface.router, prefix="/surface", tags=["Volatility Surface"])
api_router.include_router(market.router, prefix="/market", tags=["Market"])
api_router.include_router(smile.router, prefix="/smile", tags=["Volatility Smile"])
