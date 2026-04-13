"""Tests for ProxyEngine DistServe delegation."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import JSONResponse

from dlrouter.core.proxy_engine import ProxyEngine


class TestHandleDistServe:
    @pytest.mark.asyncio
    async def test_delegates_pd_request_to_backend_with_context(self):
        backend = MagicMock()
        backend.handle_pd_request = AsyncMock(return_value='ok')
        node_manager = MagicMock()
        node_manager.backend = backend
        discovery = object()

        engine = ProxyEngine(node_manager, discovery)

        response = await engine.handle_distserve(
            {'messages': []},
            'model-a',
            '/v1/chat/completions',
            stream=True,
            request_key='tenant-1',
        )

        assert response == 'ok'
        backend.handle_pd_request.assert_awaited_once()
        call = backend.handle_pd_request.await_args
        assert call.args[:4] == (
            {'messages': []},
            'model-a',
            '/v1/chat/completions',
            True,
        )
        context = call.args[4]
        assert context.node_manager is node_manager
        assert context.service_discovery is discovery
        assert context.request_key == 'tenant-1'

    @pytest.mark.asyncio
    async def test_returns_400_when_backend_does_not_implement_pd(self):
        backend = MagicMock()
        backend.handle_pd_request = AsyncMock(side_effect=NotImplementedError)
        node_manager = MagicMock()
        node_manager.backend = backend

        engine = ProxyEngine(node_manager)

        response = await engine.handle_distserve(
            {'messages': []},
            'model-a',
            '/v1/chat/completions',
        )

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        assert json.loads(response.body) == {'error': 'Current backend does not support PD disaggregation'}
