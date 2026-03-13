"""Tests for VLLMBackend."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from dlrouter.backends.factory import create_backend
from dlrouter.backends.vllm_backend import VLLMBackend
from dlrouter.config import BackendConfig
from dlrouter.constants import BackendType


NODE_URL = 'http://10.0.0.1:8000'


class _AsyncLines:
    """Async iterable over lines of bytes, for mocking resp.content."""

    def __init__(self, body: bytes) -> None:
        self._lines = body.splitlines()

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line


def _make_session_mock(status: int = 200, body: bytes = b'', exception=None):
    """Build a mock aiohttp.ClientSession context manager chain.

    Returns (session_mock, response_mock) so callers can inspect them.
    """
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body.decode())
    resp.content = _AsyncLines(body)

    req_ctx = AsyncMock()
    if exception:
        req_ctx.__aenter__ = AsyncMock(side_effect=exception)
    else:
        req_ctx.__aenter__ = AsyncMock(return_value=resp)
    req_ctx.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.post = MagicMock(return_value=req_ctx)
    session.get = MagicMock(return_value=req_ctx)

    sess_ctx = AsyncMock()
    sess_ctx.__aenter__ = AsyncMock(return_value=session)
    sess_ctx.__aexit__ = AsyncMock(return_value=False)

    return sess_ctx, resp


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_vllm_backend(self):
        cfg = BackendConfig(type=BackendType.VLLM)
        backend = create_backend(cfg)
        assert isinstance(backend, VLLMBackend)

    def test_does_not_support_pd_disagg(self):
        cfg = BackendConfig(type=BackendType.VLLM)
        backend = create_backend(cfg)
        assert backend.supports_pd_disagg() is False


# ---------------------------------------------------------------------------
# forward_request
# ---------------------------------------------------------------------------


class TestForwardRequest:
    async def test_success(self):
        sess_ctx, _ = _make_session_mock(status=200, body=b'{"choices":[]}')
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            result = await backend.forward_request(NODE_URL, '/v1/chat/completions', {'model': 'x'})
        assert result == '{"choices":[]}'

    async def test_raises_on_connection_error(self):
        sess_ctx, _ = _make_session_mock(exception=aiohttp.ClientConnectionError('refused'))
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            with pytest.raises(aiohttp.ClientConnectionError):
                await backend.forward_request(NODE_URL, '/v1/chat/completions', {})


# ---------------------------------------------------------------------------
# stream_forward
# ---------------------------------------------------------------------------


class TestStreamForward:
    async def test_yields_non_empty_lines(self):
        body = b'data: {"id":1}\ndata: [DONE]'
        sess_ctx, _ = _make_session_mock(status=200, body=body)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            chunks = [chunk async for chunk in backend.stream_forward(NODE_URL, '/v1/chat/completions', {})]
        assert len(chunks) > 0
        combined = b''.join(chunks)
        assert b'data: {"id":1}' in combined

    async def test_raises_on_error(self):
        sess_ctx, _ = _make_session_mock(exception=aiohttp.ServerConnectionError('server error'))
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            with pytest.raises(aiohttp.ServerConnectionError):
                async for _ in backend.stream_forward(NODE_URL, '/v1/chat/completions', {}):
                    pass


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


class TestFetchModels:
    def test_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'data': [
                {'id': '/models/Qwen3-32B'},
                {'id': '/models/Llama-3'},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch('dlrouter.backends.vllm_backend.requests.get', return_value=mock_resp):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == ['/models/Qwen3-32B', '/models/Llama-3']

    def test_empty_data(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'data': []}
        mock_resp.raise_for_status = MagicMock()

        with patch('dlrouter.backends.vllm_backend.requests.get', return_value=mock_resp):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == []

    def test_connection_error_returns_empty(self):
        with patch(
            'dlrouter.backends.vllm_backend.requests.get',
            side_effect=Exception('connection refused'),
        ):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == []


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    async def test_healthy_200(self):
        sess_ctx, _ = _make_session_mock(status=200)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            assert await backend.check_health(NODE_URL) is True

    async def test_unhealthy_non_200(self):
        sess_ctx, _ = _make_session_mock(status=503)
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            assert await backend.check_health(NODE_URL) is False

    async def test_connection_error_returns_false(self):
        sess_ctx, _ = _make_session_mock(exception=aiohttp.ClientConnectionError('refused'))
        with patch('aiohttp.ClientSession', return_value=sess_ctx):
            backend = VLLMBackend()
            assert await backend.check_health(NODE_URL) is False


# ---------------------------------------------------------------------------
# deregister_node
# ---------------------------------------------------------------------------


class TestDeregisterNode:
    def test_is_noop(self):
        backend = VLLMBackend()
        # Should not raise
        backend.deregister_node(NODE_URL)
