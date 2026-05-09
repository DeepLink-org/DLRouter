"""Helpers for DP-aware logical node URLs."""

from __future__ import annotations


def parse_dp_rank(url: str) -> int | None:
    """Return the DP rank encoded as a trailing ``@rank`` suffix, if present."""
    if '@' not in url:
        return None

    _, suffix = url.rsplit('@', 1)
    if not suffix.isdigit():
        return None
    return int(suffix)


def normalize_dp_aware_url(url: str) -> str:
    """Strip a trailing numeric ``@rank`` suffix from a logical node URL."""
    if parse_dp_rank(url) is None:
        return url
    base_url, _ = url.rsplit('@', 1)
    return base_url


def is_dp_aware_url(url: str) -> bool:
    """Return whether a URL uses DLRouter's DP-aware logical node suffix."""
    return parse_dp_rank(url) is not None


def expand_dp_aware_urls(urls: list[str], dp_size: int) -> list[str]:
    """Expand physical node URLs to rank-level logical node URLs."""
    if dp_size < 1:
        raise ValueError('dp_size must be >= 1')
    if dp_size == 1:
        return list(urls)

    expanded: list[str] = []
    for url in urls:
        base_url = normalize_dp_aware_url(url)
        expanded.extend(f'{base_url}@{rank}' for rank in range(dp_size))
    return expanded
