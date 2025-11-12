"""Backend configuration using environment variables."""
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Options Desk API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "REST API for options pricing, Greeks, and volatility surfaces"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # Backend dev server
        "http://localhost:8080",  # Alternative frontend port
        "https://vertex-ai-qz494-larx-785c.ue.r.appspot.com",  # App Engine frontend
        "https://backend-dot-vertex-ai-qz494-larx-785c.ue.r.appspot.com",  # App Engine backend
    ]

    # Data Configuration
    DEFAULT_RISK_FREE_RATE: float = 0.05
    DEFAULT_DATA_SOURCE: str = "yfinance"

    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 300  # 5 minutes

    # Authentication / Authorization
    ENFORCE_IAP_AUTH: bool = False
    IAP_AUDIENCE: Optional[str] = None
    IAP_EXEMPT_PATHS: List[str] = ["/health"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
