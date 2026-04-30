"""Tests for shared backend utility helpers."""

from dlrouter.backends.utils import normalize_backend_url, parse_csv_list


def test_parse_csv_list_handles_none_and_empty_strings() -> None:
    assert parse_csv_list(None) == []
    assert parse_csv_list('') == []
    assert parse_csv_list(' , , ') == []


def test_parse_csv_list_strips_whitespace_and_drops_empty_items() -> None:
    assert parse_csv_list(' model-a, model-b ,, model-c ') == [
        'model-a',
        'model-b',
        'model-c',
    ]


def test_parse_csv_list_accepts_non_string_cli_values() -> None:
    assert parse_csv_list(8998) == ['8998']


def test_normalize_backend_url_removes_trailing_slashes_by_default() -> None:
    assert normalize_backend_url('http://10.0.0.1:8000///') == 'http://10.0.0.1:8000'


def test_normalize_backend_url_can_strip_scheme_for_node_info_http_address() -> None:
    assert normalize_backend_url('https://10.0.0.1:8000/', strip_scheme=True) == '10.0.0.1:8000'
