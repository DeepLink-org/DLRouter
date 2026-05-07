"""Tests for VLLMBackend."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.factory import create_backend, get_backend_definition
from dlrouter.backends.lmdeploy import LMDeployBackend
from dlrouter.backends.vllm import (
    VLLM_BACKEND_DEFINITION,
    VLLMBackend,
    VLLMKVTransferAdapter,
    VLLMPDConfig,
)
from dlrouter.constants import BackendType, ServiceDiscoveryMode


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
    """Build a mock aiohttp.ClientSession for the persistent session pattern.

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

    # Create a mock session (not context manager, direct session object)
    session = MagicMock()
    session.post = MagicMock(return_value=req_ctx)
    session.get = MagicMock(return_value=req_ctx)
    session.closed = False

    return session, resp


async def _get_backend_with_mock_session(
    status: int = 200,
    body: bytes = b'',
    exception=None,
):
    """Create a VLLMBackend with mocked _get_session."""
    session, resp = _make_session_mock(status, body, exception)
    backend = VLLMBackend()
    backend._get_session = AsyncMock(return_value=session)
    return backend, session, resp


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_vllm_backend(self):
        backend = create_backend(BackendType.VLLM)
        assert isinstance(backend, VLLMBackend)

    def test_uses_vllm_backend_definition(self):
        backend = create_backend(BackendType.VLLM)
        definition = get_backend_definition(BackendType.VLLM)

        assert definition is VLLM_BACKEND_DEFINITION
        assert isinstance(backend, VLLMBackend)
        assert definition.supports('check_health') is True

    def test_supports_pd_disagg(self):
        backend = create_backend(BackendType.VLLM)
        assert backend.supports_pd_disagg() is True


# ---------------------------------------------------------------------------
# forward_request
# ---------------------------------------------------------------------------


class TestForwardRequest:
    async def test_success(self):
        backend, _, _ = await _get_backend_with_mock_session(
            status=200,
            body=b'{"choices":[]}',
        )
        result = await backend.forward_request(
            NODE_URL,
            '/v1/chat/completions',
            {'model': 'x'},
        )
        assert result == '{"choices":[]}'

    async def test_raises_on_connection_error(self):
        backend, _, _ = await _get_backend_with_mock_session(
            exception=aiohttp.ClientConnectionError('refused'),
        )
        with pytest.raises(aiohttp.ClientConnectionError):
            await backend.forward_request(
                NODE_URL,
                '/v1/chat/completions',
                {},
            )


# ---------------------------------------------------------------------------
# stream_forward
# ---------------------------------------------------------------------------


class TestStreamForward:
    async def test_yields_non_empty_lines(self):
        body = b'data: {"id":1}\ndata: [DONE]'
        backend, _, _ = await _get_backend_with_mock_session(
            status=200,
            body=body,
        )
        chunks = [
            chunk
            async for chunk in backend.stream_forward(
                NODE_URL,
                '/v1/chat/completions',
                {},
            )
        ]
        assert len(chunks) > 0
        combined = b''.join(chunks)
        assert b'data: {"id":1}' in combined

    async def test_raises_on_error(self):
        backend, _, _ = await _get_backend_with_mock_session(
            exception=aiohttp.ServerConnectionError('server error'),
        )
        with pytest.raises(aiohttp.ServerConnectionError):
            async for _ in backend.stream_forward(
                NODE_URL,
                '/v1/chat/completions',
                {},
            ):
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

        with patch(
            'dlrouter.backends.vllm.backend.requests.get',
            return_value=mock_resp,
        ):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == ['/models/Qwen3-32B', '/models/Llama-3']

    def test_empty_data(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {'data': []}
        mock_resp.raise_for_status = MagicMock()

        with patch(
            'dlrouter.backends.vllm.backend.requests.get',
            return_value=mock_resp,
        ):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == []

    def test_connection_error_returns_empty(self):
        with patch(
            'dlrouter.backends.vllm.backend.requests.get',
            side_effect=Exception('connection refused'),
        ):
            backend = VLLMBackend()
            models = backend.fetch_models(NODE_URL)

        assert models == []


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------


def _make_session_ctx_mock(status: int = 200, exception=None):
    """Build a mock for aiohttp.ClientSession as async context manager."""
    resp = AsyncMock()
    resp.status = status

    req_ctx = AsyncMock()
    if exception:
        req_ctx.__aenter__ = AsyncMock(side_effect=exception)
    else:
        req_ctx.__aenter__ = AsyncMock(return_value=resp)
    req_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=req_ctx)

    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    return session_ctx


class TestCheckHealth:
    async def test_healthy_200(self):
        session_ctx = _make_session_ctx_mock(status=200)
        with patch('aiohttp.ClientSession', return_value=session_ctx):
            backend = VLLMBackend()

            assert await backend.check_health(NODE_URL) is True

    async def test_unhealthy_non_200(self):
        session_ctx = _make_session_ctx_mock(status=503)
        with patch('aiohttp.ClientSession', return_value=session_ctx):
            backend = VLLMBackend()

            assert await backend.check_health(NODE_URL) is False

    async def test_connection_error_returns_false(self):
        session_ctx = _make_session_ctx_mock(
            exception=aiohttp.ClientConnectionError('refused'),
        )
        with patch('aiohttp.ClientSession', return_value=session_ctx):
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


# ---------------------------------------------------------------------------
# vLLM PD Disaggregation Support
# ---------------------------------------------------------------------------


class _AsyncChunks:
    """Async iterable over chunks of bytes, for mocking iter_chunked."""

    def __init__(self, body: bytes, chunk_size: int = 1024) -> None:
        self._chunks = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


def _make_session_mock_with_chunks(
    status: int = 200,
    body: bytes = b'',
    exception=None,
):
    """Build a mock for sessions that support iter_chunked."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body.decode())
    resp.content = MagicMock()
    resp.content.iter_chunked = MagicMock(return_value=_AsyncChunks(body))

    req_ctx = AsyncMock()
    if exception:
        req_ctx.__aenter__ = AsyncMock(side_effect=exception)
    else:
        req_ctx.__aenter__ = AsyncMock(return_value=resp)
    req_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=req_ctx)
    session.closed = False

    return session, resp


class TestForwardWithRequestId:
    async def test_success(self):
        body = b'{"choices":[]}'
        session, _ = _make_session_mock_with_chunks(status=200, body=body)
        backend = VLLMBackend()
        backend._get_session = AsyncMock(return_value=session)

        result = await backend.forward_with_request_id(
            NODE_URL,
            '/v1/chat/completions',
            {'model': 'x'},
            '___prefill_addr_p___decode_addr_d_uuid',
        )
        assert result == '{"choices":[]}'

        # Verify headers were passed
        call_args = session.post.call_args
        assert 'headers' in call_args.kwargs
        assert 'X-Request-Id' in call_args.kwargs['headers']
        assert '___prefill_addr_p' in call_args.kwargs['headers']['X-Request-Id']

    async def test_raises_on_error(self):
        session, _ = _make_session_mock_with_chunks(exception=aiohttp.ClientConnectionError('refused'))
        backend = VLLMBackend()
        backend._get_session = AsyncMock(return_value=session)

        with pytest.raises(aiohttp.ClientConnectionError):
            await backend.forward_with_request_id(
                NODE_URL,
                '/v1/chat/completions',
                {},
                'request_id',
            )


class TestStreamForwardWithRequestId:
    async def test_yields_chunks(self):
        body = b'data: {"id":1}\ndata: [DONE]'
        session, _ = _make_session_mock_with_chunks(status=200, body=body)
        backend = VLLMBackend()
        backend._get_session = AsyncMock(return_value=session)

        chunks = [
            chunk
            async for chunk in backend.stream_forward_with_request_id(
                NODE_URL,
                '/v1/chat/completions',
                {},
                '___prefill_addr_p___decode_addr_d_uuid',
            )
        ]
        assert len(chunks) > 0
        combined = b''.join(chunks)
        assert b'data:' in combined

    async def test_raises_on_error(self):
        session, _ = _make_session_mock_with_chunks(exception=aiohttp.ServerConnectionError('server error'))
        backend = VLLMBackend()
        backend._get_session = AsyncMock(return_value=session)

        with pytest.raises(aiohttp.ServerConnectionError):
            async for _ in backend.stream_forward_with_request_id(
                NODE_URL,
                '/v1/chat/completions',
                {},
                'request_id',
            ):
                pass

    async def test_request_id_header_is_set(self):
        body = b'{"choices":[]}'
        session, _ = _make_session_mock_with_chunks(status=200, body=body)
        backend = VLLMBackend()
        backend._get_session = AsyncMock(return_value=session)

        request_id = '___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30001_abc'

        async for _ in backend.stream_forward_with_request_id(
            NODE_URL,
            '/v1/chat/completions',
            {},
            request_id,
        ):
            pass

        # Verify the request_id header was set correctly
        call_args = session.post.call_args
        headers = call_args.kwargs['headers']
        assert headers['X-Request-Id'] == request_id
        assert '___prefill_addr_10.0.0.1:30001' in headers['X-Request-Id']
        assert '___decode_addr_10.0.0.2:30001' in headers['X-Request-Id']


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_get_cli_args_returns_list(self):
        args = VLLMBackend.get_cli_args()
        assert isinstance(args, list)
        assert [arg.name for arg in args] == [
            'zmq_host',
            'zmq_port',
            'zmq_ping_timeout',
            'prefill_urls',
            'decode_urls',
            'models',
        ]

    def test_cli_args_have_required_fields(self):
        args = VLLMBackend.get_cli_args()
        for arg in args:
            assert hasattr(arg, 'name')
            assert hasattr(arg, 'type')
            assert hasattr(arg, 'default')
            assert hasattr(arg, 'help')

    def test_zmq_port_arg_exists(self):
        args = VLLMBackend.get_cli_args()
        zmq_port_arg = next((a for a in args if a.name == 'zmq_port'), None)
        assert zmq_port_arg is not None
        assert zmq_port_arg.type is int
        assert zmq_port_arg.default == 30001


class TestParseConfig:
    def test_parse_config_returns_vllm_pd_config(self):
        config = VLLMBackend.parse_config(
            zmq_host='127.0.0.1',
            zmq_port=30002,
            zmq_ping_timeout=10,
            models='model-a,model-b',
        )
        assert isinstance(config, VLLMPDConfig)
        assert config.zmq_host == '127.0.0.1'
        assert config.zmq_port == 30002
        assert config.ping_timeout_seconds == 10
        assert config.models == ['model-a', 'model-b']

    def test_parse_config_defaults(self):
        config = VLLMBackend.parse_config()
        assert config.discovery_mode is ServiceDiscoveryMode.HEARTBEAT
        assert config.pd_protocol == 'two_stage_kv_transfer'
        assert config.zmq_host == '0.0.0.0'
        assert config.zmq_port == 30001
        assert config.ping_timeout_seconds == 5
        assert config.models == []

    def test_parse_config_strips_model_whitespace(self):
        config = VLLMBackend.parse_config(models='  model-a , model-b  ')
        assert config.models == ['model-a', 'model-b']

    def test_parse_config_empty_models(self):
        config = VLLMBackend.parse_config(models=None)
        assert config.models == []

    def test_parse_config_drops_empty_csv_items(self):
        config = VLLMBackend.parse_config(
            models=' model-a, , model-b, ',
            prefill_urls=' http://10.0.0.1:8200, ',
            decode_urls=' http://10.0.0.2:8200, ',
        )

        assert config.models == ['model-a', 'model-b']
        assert config.prefill_urls == ['http://10.0.0.1:8200']
        assert config.decode_urls == ['http://10.0.0.2:8200']

    def test_parse_config_supports_two_stage_protocol_fields(self):
        config = VLLMBackend.parse_config(
            pd_protocol='two_stage_kv_transfer',
            prefill_urls='http://10.0.0.1:8200',
            decode_urls='http://10.0.0.2:8200',
        )

        assert config.discovery_mode is ServiceDiscoveryMode.STATIC
        assert config.pd_protocol == 'two_stage_kv_transfer'
        assert config.prefill_urls == ['http://10.0.0.1:8200']
        assert config.decode_urls == ['http://10.0.0.2:8200']

    def test_parse_config_defaults_to_two_stage_protocol(self):
        config = VLLMBackend.parse_config()

        assert config.discovery_mode is ServiceDiscoveryMode.HEARTBEAT
        assert config.pd_protocol == 'two_stage_kv_transfer'

    def test_parse_config_infers_static_when_both_url_lists_are_present(self):
        config = VLLMBackend.parse_config(
            prefill_urls='http://10.0.0.1:8200',
            decode_urls='http://10.0.0.2:8200',
        )

        assert config.discovery_mode is ServiceDiscoveryMode.STATIC

    def test_parse_config_rejects_prefill_without_decode_urls(self):
        with pytest.raises(ValueError, match='prefill_urls and decode_urls must be provided together'):
            VLLMBackend.parse_config(prefill_urls='http://10.0.0.1:8200')

    def test_parse_config_rejects_decode_without_prefill_urls(self):
        with pytest.raises(ValueError, match='prefill_urls and decode_urls must be provided together'):
            VLLMBackend.parse_config(decode_urls='http://10.0.0.2:8200')


class TestCreate:
    def test_create_injects_pd_config_into_backend(self):
        config = VLLMPDConfig(
            discovery_mode=ServiceDiscoveryMode.STATIC,
            pd_protocol='two_stage_kv_transfer',
        )

        backend = VLLMBackend.create(config)

        assert backend.pd_config is config


# ---------------------------------------------------------------------------
# Service discovery creation
# ---------------------------------------------------------------------------


class TestCreateServiceDiscovery:
    def test_creates_zmq_discovery(self):
        from dlrouter.core.service_discovery import ZMQHeartbeatDiscovery

        backend = VLLMBackend()
        mock_node_manager = MagicMock()
        backend_config = {
            'pd_protocol': 'two_stage_kv_transfer',
            'zmq_host': '127.0.0.1',
            'zmq_port': 30002,
            'zmq_ping_timeout': 10,
            'models': 'model-a,model-b',
        }

        discovery = backend.create_service_discovery(
            ServiceDiscoveryMode.HEARTBEAT,
            backend_config,
            mock_node_manager,
        )

        assert isinstance(discovery, ZMQHeartbeatDiscovery)
        assert discovery._host == '127.0.0.1'
        assert discovery._port == 30002
        assert discovery._models == ['model-a', 'model-b']
        assert discovery._node_manager is mock_node_manager

    def test_creates_with_default_config(self):
        from dlrouter.core.service_discovery import StaticServiceDiscovery

        backend = VLLMBackend()
        mock_node_manager = MagicMock()

        discovery = backend.create_service_discovery(
            ServiceDiscoveryMode.STATIC,
            {},
            mock_node_manager,
        )

        assert isinstance(discovery, StaticServiceDiscovery)
        assert discovery._models == []
        assert discovery._initial_prefill == []
        assert discovery._initial_decode == []

    def test_creates_heartbeat_discovery_for_two_stage(self):
        backend = VLLMBackend.create(
            VLLMPDConfig(
                discovery_mode=ServiceDiscoveryMode.HEARTBEAT,
                pd_protocol='two_stage_kv_transfer',
            )
        )

        discovery = backend.create_service_discovery(
            ServiceDiscoveryMode.HEARTBEAT,
            {'pd_protocol': 'two_stage_kv_transfer'},
            MagicMock(),
        )

        assert discovery is not None

    def test_lmdeploy_backend_returns_none(self):
        backend = LMDeployBackend()
        mock_node_manager = MagicMock()

        discovery = backend.create_service_discovery(
            ServiceDiscoveryMode.HEARTBEAT,
            {},
            mock_node_manager,
        )

        assert discovery is None


# ---------------------------------------------------------------------------
# PD request handling
# ---------------------------------------------------------------------------


class TestHandlePDRequest:
    @staticmethod
    def _make_pd_pair():
        return ('http://10.0.0.1:8200', 'http://10.0.0.2:8200')

    @pytest.mark.asyncio
    async def test_handle_pd_request_no_pd_pair(self):
        """Test handle_pd_request when no P/D instances available."""
        backend = VLLMBackend()
        node_manager = MagicMock()
        node_manager.get_node_url.return_value = None

        response = await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=node_manager),
        )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_pd_request_uses_two_stage_by_default(self):
        backend = VLLMBackend.create(VLLMBackend.parse_config(discovery_mode='static'))
        backend._get_two_stage_executor = MagicMock()
        backend._get_two_stage_executor.return_value.execute = AsyncMock(return_value='ok')

        await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        backend._get_two_stage_executor.return_value.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_pd_request_dispatches_by_pd_protocol(self):
        backend = VLLMBackend.create(
            VLLMPDConfig(
                discovery_mode=ServiceDiscoveryMode.HEARTBEAT,
                pd_protocol='two_stage_kv_transfer',
            )
        )
        executor = MagicMock()
        executor.execute = AsyncMock(return_value='ok')
        backend._get_two_stage_executor = MagicMock(return_value=executor)

        result = await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        assert result == 'ok'
        backend._get_two_stage_executor.assert_called_once()
        executor.execute.assert_awaited_once()

    def test_get_two_stage_executor_caches_instance(self):
        backend = VLLMBackend.create(VLLMBackend.parse_config())
        backend._build_two_stage_executor = MagicMock(wraps=backend._build_two_stage_executor)

        first = backend._get_two_stage_executor()
        second = backend._get_two_stage_executor()

        assert first is second
        backend._build_two_stage_executor.assert_called_once()

    def test_get_pd_executor_rejects_unsupported_protocol(self):
        config = VLLMPDConfig.model_construct(
            discovery_mode=ServiceDiscoveryMode.HEARTBEAT,
            pd_protocol='unsupported_protocol',
        )
        backend = VLLMBackend.create(config)

        with pytest.raises(
            ValueError,
            match='Unsupported vLLM PD protocol: unsupported_protocol',
        ):
            backend._get_pd_executor()

    def test_build_two_stage_executor_uses_vllm_adapter(self):
        backend = VLLMBackend.create(VLLMBackend.parse_config())

        executor = backend._build_two_stage_executor()

        assert isinstance(executor.adapter, VLLMKVTransferAdapter)


class TestLegacyProtocolRemoval:
    def test_parse_config_rejects_legacy_request_id_protocol(self):
        with pytest.raises(ValidationError):
            VLLMBackend.parse_config(pd_protocol='legacy_request_id')

    @pytest.mark.asyncio
    async def test_handle_pd_request_prefill_error(self):
        """Test handle_pd_request delegates executor errors/responses."""
        backend = VLLMBackend.create(VLLMBackend.parse_config(discovery_mode='static'))
        backend._get_two_stage_executor = MagicMock()
        backend._get_two_stage_executor.return_value.execute = AsyncMock(
            return_value=JSONResponse({'error': 'Prefill phase failed'}, status_code=502)
        )

        response = await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_handle_pd_request_success_non_stream(self):
        """Test handle_pd_request returns non-stream executor response."""
        backend = VLLMBackend.create(VLLMBackend.parse_config(discovery_mode='static'))
        backend._get_two_stage_executor = MagicMock()
        backend._get_two_stage_executor.return_value.execute = AsyncMock(
            return_value=JSONResponse({'id': 'cmpl-123', 'choices': [{'text': 'hello'}]})
        )

        response = await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        assert response.status_code == 200
        backend._get_two_stage_executor.return_value.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_pd_request_success_stream(self):
        """Test handle_pd_request success (streaming)."""
        from fastapi.responses import StreamingResponse

        async def _stream():
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'

        backend = VLLMBackend.create(VLLMBackend.parse_config(discovery_mode='static'))
        backend._get_two_stage_executor = MagicMock()
        backend._get_two_stage_executor.return_value.execute = AsyncMock(
            return_value=StreamingResponse(_stream(), media_type='text/event-stream')
        )

        response = await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=True,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        assert isinstance(response, StreamingResponse)
        assert response.media_type == 'text/event-stream'

    @pytest.mark.asyncio
    async def test_handle_pd_request_decode_error_non_stream(self):
        """Test handle_pd_request surfaces decode-phase errors from executor."""
        backend = VLLMBackend.create(VLLMBackend.parse_config(discovery_mode='static'))
        backend._get_two_stage_executor = MagicMock()
        backend._get_two_stage_executor.return_value.execute = AsyncMock(
            return_value=JSONResponse({'error': 'Decode phase failed'}, status_code=502)
        )

        response = await backend.handle_pd_request(
            {'model': 'test-model', 'messages': []},
            'test-model',
            '/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        assert response.status_code == 502
