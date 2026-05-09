"""vLLM DP-aware logical URL helpers."""

from dlrouter.core.dp_url import (
    expand_dp_aware_urls,
    is_dp_aware_url,
    normalize_dp_aware_url,
    parse_dp_rank,
)


__all__ = [
    'expand_dp_aware_urls',
    'is_dp_aware_url',
    'normalize_dp_aware_url',
    'parse_dp_rank',
]
