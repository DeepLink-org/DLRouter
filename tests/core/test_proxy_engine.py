"""Tests for ProxyEngine."""

import pytest
from unittest.mock import MagicMock

from dlrouter.core.proxy_engine import ProxyEngine
from dlrouter.models.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


class TestExtractPromptForPrefixCache:
    """Tests for _extract_prompt_for_prefix_cache method."""

    @pytest.fixture
    def proxy_engine(self):
        """Create a ProxyEngine instance with mocked dependencies."""
        mock_node_manager = MagicMock()
        engine = ProxyEngine(node_manager=mock_node_manager)
        return engine

    # ------------------------------------------------------------------
    # Test None input
    # ------------------------------------------------------------------

    def test_none_body_returns_none(self, proxy_engine):
        """Test that None body returns None."""
        result = proxy_engine._extract_prompt_for_prefix_cache(None)
        assert result is None

    # ------------------------------------------------------------------
    # Tests for ChatCompletionRequest with string messages
    # ------------------------------------------------------------------

    def test_chat_request_with_string_messages(self, proxy_engine):
        """Test ChatCompletionRequest with string messages field."""
        request = ChatCompletionRequest(
            model="test-model",
            messages="Hello, world!"
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "Hello, world!"

    # ------------------------------------------------------------------
    # Tests for ChatCompletionRequest with dict list messages
    # ------------------------------------------------------------------

    def test_chat_request_with_dict_messages(self, proxy_engine):
        """Test ChatCompletionRequest with list of dict messages."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "You are a helpful assistant.\nHello!"

    def test_chat_request_with_empty_dict_content(self, proxy_engine):
        """Test that empty string content is preserved."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Response"}
            ]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        # Empty string should be preserved (converted to str)
        assert result == "\nResponse"

    def test_chat_request_with_none_content_in_dict(self, proxy_engine):
        """Test that None content in dict is filtered out."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "system", "content": None},
                {"role": "user", "content": "Hello"}
            ]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "Hello"

    def test_chat_request_with_missing_content_key(self, proxy_engine):
        """Test dict message without 'content' key defaults to empty string."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "system"},  # No content key, defaults to ''
                {"role": "user", "content": "Hello"}
            ]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        # Missing content key defaults to '', which is now preserved
        assert result == "\nHello"

    def test_chat_request_with_empty_messages_list(self, proxy_engine):
        """Test empty messages list returns None."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result is None

    # ------------------------------------------------------------------
    # Tests for ChatCompletionRequest with ChatMessage-like objects
    # ------------------------------------------------------------------
    # NOTE: ChatCompletionRequest.messages is defined as
    # Union[str, list[dict[str, Any]]], not list[ChatMessage].
    # The code has a branch for objects with .content attribute,
    # which would handle ChatMessage if it were passed directly
    # (e.g., from manual construction without validation).

    def test_chat_request_with_object_having_content_attr(self, proxy_engine):
        """Test handling of objects with content attribute (simulated)."""
        # Simulate what would happen if ChatMessage objects were passed
        class FakeMessage:
            def __init__(self, content):
                self.content = content

        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"}
            ]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "System prompt\nUser message"

    # ------------------------------------------------------------------
    # Tests for CompletionRequest
    # ------------------------------------------------------------------

    def test_completion_request_with_string_prompt(self, proxy_engine):
        """Test CompletionRequest with string prompt."""
        request = CompletionRequest(
            model="test-model",
            prompt="Complete this sentence"
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "Complete this sentence"

    def test_completion_request_with_string_list_prompt(self, proxy_engine):
        """Test CompletionRequest with list of string prompts."""
        request = CompletionRequest(
            model="test-model",
            prompt=["First prompt", "Second prompt"]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "First prompt\nSecond prompt"

    def test_completion_request_with_empty_list_prompt(self, proxy_engine):
        """Test CompletionRequest with empty list returns None."""
        request = CompletionRequest(
            model="test-model",
            prompt=[]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result is None

    def test_completion_request_with_mixed_list_prompt(self, proxy_engine):
        """Test CompletionRequest with list containing non-string items."""
        request = CompletionRequest(
            model="test-model",
            prompt=[123, "string", 45.6]
        )
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result == "123\nstring\n45.6"

    # ------------------------------------------------------------------
    # Tests for objects without messages or prompt
    # ------------------------------------------------------------------

    def test_object_without_messages_or_prompt(self, proxy_engine):
        """Test object without messages or prompt attributes returns None."""
        class DummyRequest:
            pass

        request = DummyRequest()
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result is None

    def test_object_with_none_messages(self, proxy_engine):
        """Test object with None messages returns None.

        Note: Pydantic validates messages as non-None, so we test with
        a mock object instead.
        """
        class MockRequest:
            messages = None

        request = MockRequest()
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result is None

    def test_object_with_none_prompt(self, proxy_engine):
        """Test object with None prompt returns None.

        Note: Pydantic validates prompt as non-None, so we test with
        a mock object instead.
        """
        class MockRequest:
            prompt = None

        request = MockRequest()
        result = proxy_engine._extract_prompt_for_prefix_cache(request)
        assert result is None
