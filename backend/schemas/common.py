"""Common schemas used across endpoints."""
from pydantic import BaseModel, Field
from enum import Enum


class OptionType(str, Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")
