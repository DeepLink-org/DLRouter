"""Tests for LMDeploy backend PD orchestration."""

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.factory import create_backend, get_backend_definition
from dlrouter.backends.lmdeploy import (
    LMDEPLOY_BACKEND_DEFINITION,
    LMDeployBackend,
    LMDeployPDConfig,
)
from dlrouter.constants import BackendType


async def _single_chunk_stream() -> AsyncIterator[bytes]:
    yield b'data: ok\n\n'


class TestFactory:
    def test_uses_lmdeploy_backend_definition(self):
        backend = create_backend(BackendType.LMDEPLOY)
        definition = get_backend_definition(BackendType.LMDEPLOY)

        assert definition is LMDEPLOY_BACKEND_DEFINITION
        assert isinstance(backend, LMDeployBackend)
        assert definition.supports('register_node') is True


class TestHandlePDRequest:
    @pytest.mark.asyncio
    async def test_non_stream_request_is_fully_delegated_inside_backend(self):
        backend = LMDeployBackend(LMDeployPDConfig(dummy_prefill=False))
        backend.prefill_request = AsyncMock(return_value={'id': '42'})
        backend.decode_request = AsyncMock(return_value='{"ok": true}')
        backend.connect_pd = AsyncMock()
        backend.is_connected_pd = MagicMock(return_value=False)
        backend.shelf_prefill_session = MagicMock()
        backend.unshelf_prefill_session = MagicMock()

        node_manager = MagicMock()
        node_manager.get_node_url.side_effect = [
            'http://prefill:8000',
            'http://decode:8000',
        ]
        node_manager.pre_call.side_effect = [1.0, 2.0]

        response = await backend.handle_pd_request(
            {'prompt': 'hi'},
            'model-a',
            '/v1/completions',
            False,
            PDRequestContext(node_manager=node_manager, request_key='tenant-1'),
        )

        assert isinstance(response, JSONResponse)
        assert json.loads(response.body) == {'ok': True}
        backend.prefill_request.assert_awaited_once_with(
            'http://prefill:8000',
            '/v1/completions',
            {'prompt': 'hi'},
        )
        backend.connect_pd.assert_awaited_once_with(
            'http://prefill:8000',
            'http://decode:8000',
        )
        decode_call = backend.decode_request.await_args
        assert decode_call.args[:3] == (
            'http://decode:8000',
            '/v1/completions',
            {'prompt': 'hi', '_prefill_url': 'http://prefill:8000'},
        )
        assert decode_call.args[3] == {'id': '42'}
        assert decode_call.kwargs == {'stream': False}
        backend.shelf_prefill_session.assert_called_once_with(
            'http://prefill:8000',
            'http://decode:8000',
            '42',
        )
        backend.unshelf_prefill_session.assert_called_once_with(
            'http://prefill:8000',
            'http://decode:8000',
            '42',
        )
        assert node_manager.post_call.call_count == 2

    @pytest.mark.asyncio
    async def test_streaming_cleanup_runs_in_background(self):
        backend = LMDeployBackend(LMDeployPDConfig(dummy_prefill=False))
        backend.prefill_request = AsyncMock(return_value={'id': '42'})
        backend.decode_request = AsyncMock(return_value=_single_chunk_stream())
        backend.connect_pd = AsyncMock()
        backend.is_connected_pd = MagicMock(return_value=True)
        backend.shelf_prefill_session = MagicMock()
        backend.unshelf_prefill_session = MagicMock()

        node_manager = MagicMock()
        node_manager.get_node_url.side_effect = [
            'http://prefill:8000',
            'http://decode:8000',
        ]
        node_manager.pre_call.side_effect = [1.0, 2.0]

        response = await backend.handle_pd_request(
            {'prompt': 'hi'},
            'model-a',
            '/v1/completions',
            True,
            PDRequestContext(node_manager=node_manager),
        )

        assert isinstance(response, StreamingResponse)
        backend.unshelf_prefill_session.assert_not_called()
        assert node_manager.post_call.call_count == 1

        await response.background()

        backend.unshelf_prefill_session.assert_called_once_with(
            'http://prefill:8000',
            'http://decode:8000',
            '42',
        )
        assert node_manager.post_call.call_count == 2
