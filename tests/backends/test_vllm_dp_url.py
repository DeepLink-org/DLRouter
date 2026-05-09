"""Tests for vLLM DP-aware logical URL helpers."""

from dlrouter.backends.vllm.dp_url import (
    expand_dp_aware_urls,
    is_dp_aware_url,
    normalize_dp_aware_url,
    parse_dp_rank,
)


def test_expand_dp_aware_urls_creates_ranked_logical_urls() -> None:
    assert expand_dp_aware_urls(['http://host:8000'], 3) == [
        'http://host:8000@0',
        'http://host:8000@1',
        'http://host:8000@2',
    ]


def test_expand_dp_aware_urls_leaves_urls_unchanged_for_single_rank() -> None:
    assert expand_dp_aware_urls(['http://host:8000'], 1) == ['http://host:8000']


def test_parse_dp_rank_extracts_numeric_suffix() -> None:
    assert parse_dp_rank('http://host:8000@7') == 7
    assert normalize_dp_aware_url('http://host:8000@7') == 'http://host:8000'
    assert is_dp_aware_url('http://host:8000@7') is True


def test_parse_dp_rank_supports_ipv6_hosts() -> None:
    assert parse_dp_rank('http://[::1]:8000@3') == 3
    assert normalize_dp_aware_url('http://[::1]:8000@3') == 'http://[::1]:8000'


def test_parse_dp_rank_does_not_treat_userinfo_as_rank() -> None:
    url = 'http://user:pass@host:8000'

    assert parse_dp_rank(url) is None
    assert normalize_dp_aware_url(url) == url
    assert is_dp_aware_url(url) is False


def test_invalid_rank_suffix_is_treated_as_plain_url() -> None:
    url = 'http://host:8000@abc'

    assert parse_dp_rank(url) is None
    assert normalize_dp_aware_url(url) == url
    assert is_dp_aware_url(url) is False
