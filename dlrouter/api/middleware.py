"""Middleware for DLRouter API."""

from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


security = HTTPBearer(auto_error=False)

# Module-level API keys storage
_api_keys: Optional[list[str]] = None


def set_api_keys(
    keys: Optional[list[str]],
) -> None:
    """Configure API keys for authentication."""
    global _api_keys
    _api_keys = keys


async def check_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> None:
    """Validate API key if configured.

    Raises:
        HTTPException: If key is invalid or missing.
    """
    if _api_keys is None:
        return
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail='API key required.',
        )
    if credentials.credentials not in _api_keys:
        raise HTTPException(
            status_code=403,
            detail='Invalid API key.',
        )
