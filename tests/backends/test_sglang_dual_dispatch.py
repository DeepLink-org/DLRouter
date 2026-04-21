"""Tests for the SGLang dual-dispatch PD executor."""

import json
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.sglang.dual_dispatch import SGLangDualDispatchExecutor
from dlrouter.backends.sglang.transfer import SGLangBootstrapAdapter
from dlrouter.constants import EngineRole
from dlrouter.models.node import NodeStatus
from dlrouter.routing.round_robin import RoundRobinStrategy


def _make_node_manager():
    nodes = {
        'http://10.0.0.1:8100': NodeStatus(
            role=EngineRole.PREFILL,
            models=['qwen3-32b'],
        ),
        'http://10.0.0.2:8200': NodeStatus(
            role=EngineRole.DECODE,
            models=['qwen3-32b'],
        ),
    }
    nm = MagicMock()
    nm.nodes = nodes
    nm._lock = threading.RLock()
    nm._router = RoundRobinStrategy()
    nm.prefill_nodes = {url: st for url, st in nodes.items() if st.role == EngineRole.PREFILL}
    nm.decode_nodes = {url: st for url, st in nodes.items() if st.role == EngineRole.DECODE}
    nm.pre_call.return_value = 1000.0

    def get_node_url(model_name: str, role: EngineRole = EngineRole.HYBRID, request_key: str | None = None):
        candidates = {url: st for url, st in nodes.items() if st.role == role}
        return nm._router.select_node(model_name, candidates, request_key)

    nm.get_node_url.side_effect = get_node_url
    return nm


def _make_empty_node_manager():
    nm = MagicMock()
    nm.nodes = {}
    nm._lock = threading.RLock()
    nm._router = RoundRobinStrategy()
    nm.prefill_nodes = {}
    nm.decode_nodes = {}
    nm.get_node_url.return_value = None
    return nm


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_execute_non_stream_dual_dispatches_same_routed_request_and_returns_decode_json():
    backend = MagicMock()
    backend.forward_request = AsyncMock(
        side_effect=[
            '{"prefill": true}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    executor = SGLangDualDispatchExecutor(
        backend=backend,
        adapter=SGLangBootstrapAdapter(
            {'http://10.0.0.1:8100': 8998},
            room_generator=lambda: 777,
        ),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=_make_node_manager()),
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        'id': 'cmpl-1',
        'choices': [{'text': 'hello'}],
    }
    assert backend.forward_request.await_count == 2

    prefill_call = backend.forward_request.await_args_list[0].args
    decode_call = backend.forward_request.await_args_list[1].args

    assert prefill_call[0] == 'http://10.0.0.1:8100'
    assert decode_call[0] == 'http://10.0.0.2:8200'
    assert prefill_call[1] == '/v1/chat/completions'
    assert decode_call[1] == '/v1/chat/completions'
    assert prefill_call[2] == decode_call[2]
    assert prefill_call[2]['bootstrap_host'] == '10.0.0.1'
    assert prefill_call[2]['bootstrap_port'] == 8998
    assert prefill_call[2]['bootstrap_room'] == 777


@pytest.mark.asyncio
async def test_execute_non_stream_returns_503_when_no_pd_pair():
    backend = MagicMock()
    executor = SGLangDualDispatchExecutor(
        backend=backend,
        adapter=SGLangBootstrapAdapter({}),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b'},
        endpoint='/v1/completions',
        stream=False,
        context=PDRequestContext(node_manager=_make_empty_node_manager()),
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_execute_stream_returns_decode_stream_and_drains_prefill():
    backend = MagicMock()
    backend.forward_request = AsyncMock(return_value='{"prefill": true}')
    backend.stream_forward = MagicMock(return_value=_aiter([b'data: {"choices": []}\n\n', b'data: [DONE]\n\n']))
    executor = SGLangDualDispatchExecutor(
        backend=backend,
        adapter=SGLangBootstrapAdapter(
            {'http://10.0.0.1:8100': 8998},
            room_generator=lambda: 888,
        ),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': [], 'stream': True},
        endpoint='/v1/chat/completions',
        stream=True,
        context=PDRequestContext(node_manager=_make_node_manager()),
    )

    assert isinstance(response, StreamingResponse)

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert chunks == [b'data: {"choices": []}\n\n', b'data: [DONE]\n\n']
    backend.forward_request.assert_awaited_once()
    backend.stream_forward.assert_called_once()


@pytest.mark.asyncio
async def test_execute_non_stream_calls_pre_call_and_post_call_for_both_nodes():
    backend = MagicMock()
    backend.forward_request = AsyncMock(
        side_effect=[
            '{"prefill": true}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = SGLangDualDispatchExecutor(
        backend=backend,
        adapter=SGLangBootstrapAdapter({'http://10.0.0.1:8100': 8998}),
    )

    await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert node_manager.pre_call.call_count == 2
    assert node_manager.post_call.call_count == 2
    assert node_manager.pre_call.call_args_list[0].args[0] == 'http://10.0.0.1:8100'
    assert node_manager.pre_call.call_args_list[1].args[0] == 'http://10.0.0.2:8200'
