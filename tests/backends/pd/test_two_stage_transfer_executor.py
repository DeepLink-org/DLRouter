"""Tests for the shared two-stage transfer PD executor."""

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.pd import PDPair, TwoStageTransferExecutor
from dlrouter.backends.vllm.kv_transfer import VLLMKVTransferAdapter
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

    def get_node_url(model_name: str, role: EngineRole = EngineRole.HYBRID, request_key: str | None = None):
        candidates = {url: st for url, st in nodes.items() if st.role == role}
        return nm._router.select_node(model_name, candidates, request_key)

    nm.get_node_url.side_effect = get_node_url
    return nm


def _make_empty_node_manager():
    """Build a mock NodeManager with no nodes."""
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


class _RecordingKVTransferAdapter(VLLMKVTransferAdapter):
    """Record the abort ids passed to the prefill payload builder."""

    def __init__(self) -> None:
        self.prefill_abort_ids: list[str] | None = None

    def build_prefill_request(
        self,
        request_data: dict[str, Any],
        request_id: str,
        aborted_request_ids: list[str],
    ) -> dict[str, Any]:
        self.prefill_abort_ids = list(aborted_request_ids)
        return super().build_prefill_request(
            request_data,
            request_id,
            aborted_request_ids,
        )


@pytest.mark.asyncio
async def test_execute_non_stream_success_returns_json_response():
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 200
    assert transport.forward_with_request_id.await_count == 2


@pytest.mark.asyncio
async def test_execute_starts_prefill_without_prior_abort_ids():
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9"}}',
            '{"id": "cmpl-1", "choices": [{"text": "hi"}]}',
        ]
    )
    adapter = _RecordingKVTransferAdapter()
    node_manager = _make_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
        adapter=adapter,
    )

    await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert adapter.prefill_abort_ids == []


@pytest.mark.asyncio
async def test_execute_non_stream_uses_same_encoded_request_id_for_prefill_and_decode():
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 200
    assert transport.forward_with_request_id.await_count == 2

    first_request_id = transport.forward_with_request_id.await_args_list[0].args[3]
    second_request_id = transport.forward_with_request_id.await_args_list[1].args[3]

    assert first_request_id == second_request_id
    assert first_request_id.startswith('___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30002_')


@pytest.mark.asyncio
async def test_execute_passes_request_key_to_pair_selector():
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    pair_selector = MagicMock()
    pair_selector.select_pair.return_value = PDPair('http://10.0.0.1:13700', 'http://10.0.0.2:13701')
    node_manager = _make_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
        adapter=VLLMKVTransferAdapter(),
        pair_selector=pair_selector,
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(
            node_manager=node_manager,
            request_key='session-123',
        ),
    )

    assert response.status_code == 200
    pair_selector.select_pair.assert_called_once_with(
        node_manager=node_manager,
        model_name='qwen3-32b',
        request_key='session-123',
    )


@pytest.mark.asyncio
async def test_execute_non_stream_continues_without_transfer_context():
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": null}',
            '{"id": "cmpl-1", "choices": [{"text": "hello"}]}',
        ]
    )
    node_manager = _make_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert response.status_code == 200
    assert transport.forward_with_request_id.await_count == 2

    decode_request = transport.forward_with_request_id.await_args_list[1].args[2]
    assert decode_request == {'model': 'qwen3-32b', 'messages': []}


@pytest.mark.asyncio
async def test_execute_non_stream_returns_503_when_no_pd_pair():
    transport = MagicMock()
    node_manager = _make_empty_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
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
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        return_value='{"kv_transfer_params": {"remote_host": "10.0.0.9", "remote_port": 20001}}'
    )
    transport.stream_forward_with_request_id = MagicMock(
        return_value=_aiter([b'data: {"choices":[{"delta":{"content":"hello"},"stop_reason":null}]}\n\n'])
    )
    node_manager = _make_node_manager()
    executor = TwoStageTransferExecutor(
        transport=transport,
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
    transport = MagicMock()
    transport.forward_with_request_id = AsyncMock(
        side_effect=[
            '{"kv_transfer_params": {"remote_host": "10.0.0.9"}}',
            '{"id": "cmpl-1", "choices": [{"text": "hi"}]}',
        ]
    )

    node_manager = _make_node_manager()

    executor = TwoStageTransferExecutor(
        transport=transport,
        adapter=VLLMKVTransferAdapter(),
    )

    await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=node_manager),
    )

    assert node_manager.pre_call.call_count == 2
    prefill_url = node_manager.pre_call.call_args_list[0].args[0]
    decode_url = node_manager.pre_call.call_args_list[1].args[0]
    assert prefill_url == 'http://10.0.0.1:13700'
    assert decode_url == 'http://10.0.0.2:13701'

    assert node_manager.post_call.call_count == 2
    assert node_manager.post_call.call_args_list[0].args[0] == 'http://10.0.0.1:13700'
    assert node_manager.post_call.call_args_list[1].args[0] == 'http://10.0.0.2:13701'
