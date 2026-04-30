"""Tests for SGLangBackend."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.factory import create_backend, get_backend_definition
from dlrouter.backends.pd import DualDispatchExecutor
from dlrouter.backends.sglang import (
    SGLANG_BACKEND_DEFINITION,
    SGLangBackend,
    SGLangPDConfig,
)
from dlrouter.backends.sglang.bootstrap import SGLangBootstrapAdapter
from dlrouter.constants import BackendType, ServiceDiscoveryMode


class _AsyncLines:
    """Async iterable over lines of bytes, for mocking resp.content."""

    def __init__(self, body: bytes) -> None:
        self._lines = body.splitlines()

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for line in self._lines:
            yield line


def _make_session_mock(body: bytes = b'{"ok": true}'):
    """Build a mock aiohttp session with a single POST response."""
    resp = AsyncMock()
    resp.status = 200
    resp.text = AsyncMock(return_value=body.decode())
    resp.content = _AsyncLines(body)

    req_ctx = AsyncMock()
    req_ctx.__aenter__ = AsyncMock(return_value=resp)
    req_ctx.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.post = MagicMock(return_value=req_ctx)
    session.closed = False
    return session


class TestFactory:
    def test_factory_creates_sglang_backend(self):
        backend = create_backend(BackendType.SGLANG)

        assert isinstance(backend, SGLangBackend)

    def test_factory_uses_sglang_definition(self):
        definition = get_backend_definition(BackendType.SGLANG)

        assert definition is SGLANG_BACKEND_DEFINITION
        assert definition.name == 'sglang'


class TestParseConfig:
    def test_default_config_allows_backend_creation_without_static_urls(self):
        config = SGLangBackend.parse_config()

        assert config.discovery_mode is ServiceDiscoveryMode.STATIC
        assert config.prefill_urls == []
        assert config.decode_urls == []
        assert config.prefill_bootstrap_ports == []

    def test_parse_config_infers_static_from_prefill_and_decode_urls(self):
        config = SGLangBackend.parse_config(
            prefill_urls='http://10.0.0.1:8100',
            decode_urls='http://10.0.0.2:8200',
            models='Qwen3-4B',
        )

        assert config.discovery_mode is ServiceDiscoveryMode.STATIC
        assert config.prefill_urls == ['http://10.0.0.1:8100']
        assert config.decode_urls == ['http://10.0.0.2:8200']
        assert config.prefill_bootstrap_ports == [8998]
        assert config.models == ['Qwen3-4B']

    def test_parse_config_parses_explicit_bootstrap_ports(self):
        config = SGLangBackend.parse_config(
            prefill_urls='http://10.0.0.1:8100,http://10.0.0.2:8100',
            decode_urls='http://10.0.0.3:8200',
            prefill_bootstrap_ports='8998,8999',
        )

        assert config.prefill_bootstrap_ports == [8998, 8999]

    def test_parse_config_drops_empty_csv_items(self):
        config = SGLangBackend.parse_config(
            prefill_urls=' http://10.0.0.1:8100, ',
            decode_urls=' http://10.0.0.2:8200, ',
            prefill_bootstrap_ports=' 8998, ',
            models=' Qwen3-4B, , Qwen3-8B ',
        )

        assert config.prefill_urls == ['http://10.0.0.1:8100']
        assert config.decode_urls == ['http://10.0.0.2:8200']
        assert config.prefill_bootstrap_ports == [8998]
        assert config.models == ['Qwen3-4B', 'Qwen3-8B']

    def test_parse_config_rejects_missing_decode_urls(self):
        with pytest.raises(
            ValueError,
            match='prefill_urls and decode_urls must be provided together',
        ):
            SGLangBackend.parse_config(prefill_urls='http://10.0.0.1:8100')

    def test_parse_config_rejects_missing_prefill_urls(self):
        with pytest.raises(
            ValueError,
            match='prefill_urls and decode_urls must be provided together',
        ):
            SGLangBackend.parse_config(decode_urls='http://10.0.0.2:8200')

    def test_parse_config_rejects_mismatched_bootstrap_port_count(self):
        with pytest.raises(
            ValueError,
            match='prefill_bootstrap_ports must match prefill_urls length',
        ):
            SGLangBackend.parse_config(
                prefill_urls='http://10.0.0.1:8100,http://10.0.0.2:8100',
                decode_urls='http://10.0.0.3:8200',
                prefill_bootstrap_ports='8998',
            )


class TestCreateServiceDiscovery:
    def test_create_service_discovery_requires_static_urls(self):
        backend = SGLangBackend()

        with pytest.raises(
            ValueError,
            match='SGLang backend currently requires static prefill_urls and decode_urls',
        ):
            backend.create_service_discovery(
                ServiceDiscoveryMode.STATIC,
                {},
                MagicMock(),
            )

    def test_create_service_discovery_registers_static_prefill_and_decode_nodes(self):
        node_manager = MagicMock()
        backend = SGLangBackend()

        discovery = backend.create_service_discovery(
            ServiceDiscoveryMode.STATIC,
            {
                'prefill_urls': 'http://10.0.0.1:8100',
                'decode_urls': 'http://10.0.0.2:8200',
                'models': 'Qwen3-4B',
            },
            node_manager,
        )

        discovery.start()

        assert node_manager.add.call_count == 2
        prefill_url, prefill_status = node_manager.add.call_args_list[0].args
        decode_url, decode_status = node_manager.add.call_args_list[1].args
        assert prefill_url == 'http://10.0.0.1:8100'
        assert decode_url == 'http://10.0.0.2:8200'
        assert prefill_status.role.name == 'PREFILL'
        assert decode_status.role.name == 'DECODE'
        assert prefill_status.models == ['Qwen3-4B']
        assert decode_status.models == ['Qwen3-4B']


class TestForwarding:
    async def test_forward_request_posts_json_to_sglang_node(self):
        backend = SGLangBackend()
        session = _make_session_mock()
        backend._get_session = AsyncMock(return_value=session)

        result = await backend.forward_request(
            'http://10.0.0.1:8100',
            '/v1/chat/completions',
            {'model': 'Qwen3-4B'},
        )

        assert result == '{"ok": true}'
        session.post.assert_called_once()

    def test_fetch_models_returns_openai_model_ids(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {'data': [{'id': 'Qwen3-4B'}]}

        with patch(
            'dlrouter.backends.sglang.backend.requests.get',
            return_value=response,
        ):
            assert SGLangBackend().fetch_models('http://10.0.0.1:8100') == ['Qwen3-4B']

    @pytest.mark.asyncio
    async def test_handle_pd_request_uses_cached_dual_dispatch_executor(self):
        backend = SGLangBackend(
            SGLangPDConfig(
                prefill_urls=['http://10.0.0.1:8100'],
                decode_urls=['http://10.0.0.2:8200'],
                prefill_bootstrap_ports=[8998],
                models=['Qwen3-4B'],
            )
        )
        executor = MagicMock()
        executor.execute = AsyncMock(return_value='ok')
        backend._get_dual_dispatch_executor = MagicMock(return_value=executor)

        result = await backend.handle_pd_request(
            request_data={'model': 'Qwen3-4B'},
            model_name='Qwen3-4B',
            endpoint='/v1/chat/completions',
            stream=False,
            context=PDRequestContext(node_manager=MagicMock()),
        )

        assert result == 'ok'
        backend._get_dual_dispatch_executor.assert_called_once()
        executor.execute.assert_awaited_once()

    def test_get_dual_dispatch_executor_caches_instance(self):
        backend = SGLangBackend(
            SGLangPDConfig(
                prefill_urls=['http://10.0.0.1:8100'],
                decode_urls=['http://10.0.0.2:8200'],
                prefill_bootstrap_ports=[8998],
                models=['Qwen3-4B'],
            )
        )
        backend._build_dual_dispatch_executor = MagicMock(wraps=backend._build_dual_dispatch_executor)

        first = backend._get_dual_dispatch_executor()
        second = backend._get_dual_dispatch_executor()

        assert first is second
        backend._build_dual_dispatch_executor.assert_called_once()

    def test_build_dual_dispatch_executor_wires_transport_and_bootstrap_adapter(self):
        backend = SGLangBackend(
            SGLangPDConfig(
                prefill_urls=['http://10.0.0.1:8100'],
                decode_urls=['http://10.0.0.2:8200'],
                prefill_bootstrap_ports=[8998],
                models=['Qwen3-4B'],
            )
        )

        executor = backend._build_dual_dispatch_executor()

        assert isinstance(executor, DualDispatchExecutor)
        assert executor.transport is backend
        assert isinstance(executor.adapter, SGLangBootstrapAdapter)
        assert executor.adapter.bootstrap_ports_by_url == {'http://10.0.0.1:8100': 8998}
