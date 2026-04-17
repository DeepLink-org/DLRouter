"""Tests for the vLLM two-stage PD executor."""

import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.vllm.kv_transfer import VLLMKVTransferAdapter
from dlrouter.backends.vllm.two_stage import VLLMTwoStagePDExecutor
from dlrouter.constants import EngineRole
from dlrouter.models.node import NodeStatus
from dlrouter.routing.round_robin import RoundRobinStrategy


def _make_node_manager():
    """Build a mock NodeManager with P/D nodes registered."""
    nodes = {
        'http://10.0.0.1:13700': NodeStatus(
            role=EngineRole.PREFILL,
            models=['qwen3-32b'],
            zmq_address='10.0.0.1:30001',
        ),
        'http://10.0.0.2:13701': NodeStatus(
            role=EngineRole.DECODE,
            models=['qwen3-32b'],
            zmq_address='10.0.0.2:30002',
        ),
    }
    nm = MagicMock()
    nm.nodes = nodes
    nm._lock = threading.RLock()
    nm._router = RoundRobinStrategy()
    nm.prefill_nodes = {url: st for url, st in nodes.items() if st.role == EngineRole.PREFILL}
    nm.decode_nodes = {url: st for url, st in nodes.items() if st.role == EngineRole.DECODE}
    nm.pre_call.return_value = 1000.0
    return nm


def _make_empty_node_manager():
    """Build a mock NodeManager with no nodes."""
    nm = MagicMock()
    nm.nodes = {}
    nm._lock = threading.RLock()
    nm._router = RoundRobinStrategy()
    nm.prefill_nodes = {}
    nm.decode_nodes = {}
    return nm


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_execute_non_stream_success_returns_json_response():
    backend = MagicMock()
    backend.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 200
    assert backend.forward_with_request_id.await_count == 2


@pytest.mark.asyncio
async def test_execute_non_stream_uses_same_encoded_request_id_for_prefill_and_decode():
    backend = MagicMock()
    backend.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 200
    assert backend.forward_with_request_id.await_count == 2

    first_request_id = backend.forward_with_request_id.await_args_list[0].args[3]
    second_request_id = backend.forward_with_request_id.await_args_list[1].args[3]

    assert first_request_id == second_request_id
    assert first_request_id.startswith('___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30002_')


@pytest.mark.asyncio
async def test_execute_non_stream_continues_without_transfer_context():
    backend = MagicMock()
    backend.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": null}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 200
    assert backend.forward_with_request_id.await_count == 2

    decode_request = backend.forward_with_request_id.await_args_list[1].args[2]
    assert decode_request == {'model': 'qwen3-32b', 'messages': []}


@pytest.mark.asyncio
async def test_execute_non_stream_returns_503_when_no_pd_pair():
    backend = MagicMock()
    node_manager = _make_empty_node_manager()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b'},
        endpoint='/v1/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_execute_stream_returns_streaming_response():
    backend = MagicMock()
    backend.forward_with_request_id = AsyncMock(
        return_value='{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}'
    )
    backend.stream_forward_with_request_id = MagicMock(
        return_value=_aiter([b'data: {"choices":[{"delta":{"content":"hello"},"stop_reason":null}]}\n\n'])
    )
    node_manager = _make_node_manager()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': [], 'stream': True},
        endpoint='/v1/chat/completions',
        stream=True,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert isinstance(response, StreamingResponse)


@pytest.mark.asyncio
async def test_execute_non_stream_calls_pre_call_and_post_call():
    """Verify pre_call/post_call are invoked for both prefill and decode phases."""
    backend = MagicMock()
    backend.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9"}}',
            '{"id": "cmpl-1", "choices": [{"text": "hi"}]}',
        ]
    )

    node_manager = _make_node_manager()

    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    # pre_call called for prefill and decode
    assert node_manager.pre_call.call_count == 2
    prefill_url = node_manager.pre_call.call_args_list[0].args[0]
    decode_url = node_manager.pre_call.call_args_list[1].args[0]
    assert prefill_url == 'http://10.0.0.1:13700'
    assert decode_url == 'http://10.0.0.2:13701'

    # post_call called for prefill and decode
    assert node_manager.post_call.call_count == 2
    assert node_manager.post_call.call_args_list[0].args[0] == 'http://10.0.0.1:13700'
    assert node_manager.post_call.call_args_list[1].args[0] == 'http://10.0.0.2:13701'
