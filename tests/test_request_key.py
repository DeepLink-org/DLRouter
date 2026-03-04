"""Tests for request key extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dlrouter.models.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    SessionParams,
)
from dlrouter.utils.request_key import (
    REQUEST_KEY_HEADERS,
    extract_request_key,
    extract_request_key_from_body,
    extract_request_key_from_headers,
)


def _make_mock_request(headers: dict[str, str] | None = None) -> MagicMock:
    """Create a mock Starlette Request with given headers."""
    mock_request = MagicMock()
    mock_request.headers = headers or {}
    return mock_request


class TestExtractRequestKeyFromHeaders:
    """Tests for header-based request key extraction."""

    def test_no_request(self):
        """Should return None when request is None."""
        assert extract_request_key_from_headers(None) is None

    def test_empty_headers(self):
        """Should return None when no relevant headers present."""
        req = _make_mock_request({})
        assert extract_request_key_from_headers(req) is None

    @pytest.mark.parametrize(
        'header_name',
        REQUEST_KEY_HEADERS,
    )
    def test_each_header(self, header_name: str):
        """Should extract value from each supported header."""
        req = _make_mock_request({header_name: 'test-value-123'})
        assert extract_request_key_from_headers(req) == 'test-value-123'

    def test_header_priority(self):
        """Should respect header priority order."""
        # x-session-id has highest priority
        req = _make_mock_request(
            {
                'x-user-id': 'user-456',
                'x-session-id': 'session-123',
                'x-tenant-id': 'tenant-789',
            }
        )
        assert extract_request_key_from_headers(req) == 'session-123'

    def test_fallback_to_lower_priority(self):
        """Should fallback to lower priority headers."""
        req = _make_mock_request(
            {
                'x-tenant-id': 'tenant-789',
                'x-trace-id': 'trace-000',
            }
        )
        # x-tenant-id has priority 3, x-trace-id has priority 6
        assert extract_request_key_from_headers(req) == 'tenant-789'

    def test_empty_header_value_skipped(self):
        """Should skip headers with empty values."""
        req = _make_mock_request(
            {
                'x-session-id': '',
                'x-user-id': 'user-456',
            }
        )
        assert extract_request_key_from_headers(req) == 'user-456'


class TestExtractRequestKeyFromBody:
    """Tests for body-based request key extraction."""

    def test_session_params_session_id(self):
        """Should extract session_params.session_id first."""
        req = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            session_params=SessionParams(session_id='sess-123'),
            user='user-456',
        )
        assert extract_request_key_from_body(req) == 'sess-123'

    def test_user_field(self):
        """Should extract user field (OpenAI format)."""
        req = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            user='user-456',
        )
        assert extract_request_key_from_body(req) == 'user-456'

    def test_legacy_session_id(self):
        """Should extract legacy session_id field."""
        req = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            session_id='legacy-sess-789',
        )
        assert extract_request_key_from_body(req) == 'legacy-sess-789'

    def test_legacy_user_id(self):
        """Should extract legacy user_id field."""
        req = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            user_id='legacy-user-000',
        )
        assert extract_request_key_from_body(req) == 'legacy-user-000'

    def test_body_priority(self):
        """Should respect body field priority order."""
        # session_params.session_id > user > session_id > user_id
        req = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            session_params=SessionParams(session_id='sp-sess'),
            user='openai-user',
            session_id='legacy-sess',
            user_id='legacy-user',
        )
        assert extract_request_key_from_body(req) == 'sp-sess'

    def test_completion_request(self):
        """Should work with CompletionRequest."""
        req = CompletionRequest(
            model='test',
            prompt='hello',
            user='completion-user',
        )
        assert extract_request_key_from_body(req) == 'completion-user'

    def test_dict_body(self):
        """Should work with dict body."""
        body = {
            'model': 'test',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'session_params': {'session_id': 'dict-sess'},
            'user': 'dict-user',
        }
        assert extract_request_key_from_body(body) == 'dict-sess'

    def test_dict_body_user_fallback(self):
        """Should fallback to user in dict body."""
        body = {
            'model': 'test',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'user': 'dict-user',
        }
        assert extract_request_key_from_body(body) == 'dict-user'

    def test_no_key_in_body(self):
        """Should return None when no key fields present."""
        req = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        assert extract_request_key_from_body(req) is None


class TestExtractRequestKey:
    """Tests for the main extract_request_key function."""

    def test_header_takes_priority(self):
        """Headers should take priority over body fields."""
        mock_req = _make_mock_request({'x-session-id': 'header-sess'})
        body = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            user='body-user',
        )
        assert extract_request_key(mock_req, body) == 'header-sess'

    def test_body_fallback(self):
        """Should fallback to body when no headers."""
        mock_req = _make_mock_request({})
        body = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            user='body-user',
        )
        assert extract_request_key(mock_req, body) == 'body-user'

    def test_hash_fallback(self):
        """Should fallback to body hash when no explicit key."""
        mock_req = _make_mock_request({})
        body = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        result = extract_request_key(mock_req, body)
        # Should be a valid MD5 hash (32 hex chars)
        assert result is not None
        assert len(result) == 32
        assert all(c in '0123456789abcdef' for c in result)

    def test_hash_deterministic(self):
        """Hash fallback should be deterministic."""
        mock_req = _make_mock_request({})
        body = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hello world'}],
            temperature=0.7,
        )
        result1 = extract_request_key(mock_req, body)
        result2 = extract_request_key(mock_req, body)
        assert result1 == result2

    def test_disable_hash_fallback(self):
        """Should return None when hash fallback disabled."""
        mock_req = _make_mock_request({})
        body = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        result = extract_request_key(mock_req, body, fallback_to_hash=False)
        assert result is None

    def test_none_request_and_body(self):
        """Should return None when both request and body are None."""
        assert extract_request_key(None, None) is None

    def test_none_request_with_body(self):
        """Should extract from body when request is None."""
        body = ChatCompletionRequest(
            model='test',
            messages=[{'role': 'user', 'content': 'hi'}],
            user='no-header-user',
        )
        assert extract_request_key(None, body) == 'no-header-user'


class TestHashFallbackConsistency:
    """Tests to verify hash fallback produces consistent results."""

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        body1 = {'model': 'test', 'prompt': 'hello'}
        body2 = {'model': 'test', 'prompt': 'hello'}

        hash1 = extract_request_key(None, body1)
        hash2 = extract_request_key(None, body2)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        body1 = {'model': 'test', 'prompt': 'hello'}
        body2 = {'model': 'test', 'prompt': 'world'}

        hash1 = extract_request_key(None, body1)
        hash2 = extract_request_key(None, body2)
        assert hash1 != hash2

    def test_key_order_independent(self):
        """Hash should be independent of key order."""
        body1 = {'a': 1, 'b': 2, 'c': 3}
        body2 = {'c': 3, 'a': 1, 'b': 2}

        hash1 = extract_request_key(None, body1)
        hash2 = extract_request_key(None, body2)
        assert hash1 == hash2
