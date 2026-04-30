"""Shared utility helpers for backend adapters."""

from typing import Any


def parse_csv_list(value: Any) -> list[str]:
    """Parse a comma-separated CLI/config value into non-empty strings."""
    if not value:
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]


def normalize_backend_url(url: str, *, strip_scheme: bool = False) -> str:
    """Normalize backend node URL strings used by backend adapters.

    This is not a general-purpose URL parser. It assumes backend node
    inputs are host[:port] or scheme://host[:port][/]. By default the
    scheme is preserved for NodeManager URL keys. Use strip_scheme=True
    when building NodeInfo.http_address, whose contract is host:port
    without a scheme.
    """
    normalized = url.rstrip('/')
    if strip_scheme:
        return normalized.removeprefix('http://').removeprefix('https://')
    return normalized
