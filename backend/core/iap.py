"""Helpers and middleware for enforcing Google IAP authentication."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional
from urllib import error as url_error
from urllib import request as url_request

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from backend.core.config import settings

_LOGGER = logging.getLogger(__name__)
_GOOGLE_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"
_METADATA_HEADER = {"Metadata-Flavor": "Google"}
_AUTH_REQUEST = google_requests.Request()


def _fetch_metadata(path: str) -> Optional[str]:
    """Fetch a value from the GCE metadata server if available."""
    url = f"{_GOOGLE_METADATA_URL}/{path}"
    req = url_request.Request(url, headers=_METADATA_HEADER)
    try:
        with url_request.urlopen(req, timeout=1) as resp:  # type: ignore[arg-type]
            return resp.read().decode("utf-8")
    except (url_error.URLError, TimeoutError):
        return None


@lru_cache(maxsize=1)
def default_iap_audience() -> Optional[str]:
    """Best-effort construction of the App Engine IAP audience string."""
    project_number = (
        os.getenv("PROJECT_NUMBER")
        or os.getenv("GOOGLE_CLOUD_PROJECT_NUMBER")
        or _fetch_metadata("project/numeric-project-id")
    )
    project_id = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or _fetch_metadata("project/project-id")
    )
    if project_number and project_id:
        return f"/projects/{project_number}/apps/{project_id}"
    return None


class IAPAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that only allows requests with a valid IAP JWT assertion."""

    header_name = "X-Goog-IAP-JWT-Assertion"

    async def dispatch(self, request: Request, call_next):
        if not settings.ENFORCE_IAP_AUTH:
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in settings.IAP_EXEMPT_PATHS):
            return await call_next(request)

        audience = settings.IAP_AUDIENCE or default_iap_audience()
        if not audience:
            _LOGGER.error("IAP enforcement enabled but no audience configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="IAP audience not configured",
            )

        assertion = request.headers.get(self.header_name)
        if not assertion:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing IAP token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            claims = id_token.verify_token(assertion, _AUTH_REQUEST, audience=audience)
        except ValueError as exc:
            _LOGGER.warning("Invalid IAP token: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid IAP token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        request.state.user_email = claims.get("email")
        request.state.user_id = claims.get("sub")
        request.state.iap_claims = claims
        return await call_next(request)
