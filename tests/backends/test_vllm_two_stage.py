"""Tests for the vLLM two-stage PD executor."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.vllm.kv_transfer import VLLMKVTransferAdapter
from dlrouter.backends.vllm.two_stage import VLLMTwoStagePDExecutor
from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo


def _make_prefill_nodes():
    return [
        NodeInfo(
            http_address='10.0.0.1:13700',
            zmq_address='10.0.0.1:30001',
            role=EngineRole.PREFILL,
            models=['qwen3-32b'],
            metadata={
                'kv_connector': 'mooncake',
                'protocol_version': 'v1',
                'endpoint_metadata': {'transport': 'rdma'},
            },
        )
    ]


def _make_decode_nodes():
    return [
        NodeInfo(
            http_address='10.0.0.2:13701',
            zmq_address='10.0.0.2:30002',
            role=EngineRole.DECODE,
            models=['qwen3-32b'],
            metadata={
                'kv_connector': 'mooncake',
                'protocol_version': 'v1',
                'endpoint_metadata': {'transport': 'rdma'},
            },
        )
    ]


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
    discovery = MagicMock()
    discovery.get_prefill_instances.return_value = _make_prefill_nodes()
    discovery.get_decode_instances.return_value = _make_decode_nodes()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=MagicMock(), service_discovery=discovery),
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
    discovery = MagicMock()
    discovery.get_prefill_instances.return_value = _make_prefill_nodes()
    discovery.get_decode_instances.return_value = _make_decode_nodes()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=MagicMock(), service_discovery=discovery),
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
    discovery = MagicMock()
    discovery.get_prefill_instances.return_value = _make_prefill_nodes()
    discovery.get_decode_instances.return_value = _make_decode_nodes()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': []},
        endpoint='/v1/chat/completions',
        stream=False,
        context=PDRequestContext(node_manager=MagicMock(), service_discovery=discovery),
    )

    assert response.status_code == 200
    assert backend.forward_with_request_id.await_count == 2

    decode_request = backend.forward_with_request_id.await_args_list[1].args[2]
    assert decode_request == {'model': 'qwen3-32b', 'messages': []}


@pytest.mark.asyncio
async def test_execute_non_stream_returns_503_when_no_pd_pair():
    backend = MagicMock()
    discovery = MagicMock()
    discovery.get_prefill_instances.return_value = []
    discovery.get_decode_instances.return_value = []
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b'},
        endpoint='/v1/completions',
        stream=False,
        context=PDRequestContext(node_manager=MagicMock(), service_discovery=discovery),
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
    discovery = MagicMock()
    discovery.get_prefill_instances.return_value = _make_prefill_nodes()
    discovery.get_decode_instances.return_value = _make_decode_nodes()
    executor = VLLMTwoStagePDExecutor(
        backend=backend,
        adapter=VLLMKVTransferAdapter(),
    )

    response = await executor.execute(
        request_data={'model': 'qwen3-32b', 'messages': [], 'stream': True},
        endpoint='/v1/chat/completions',
        stream=True,
        context=PDRequestContext(node_manager=MagicMock(), service_discovery=discovery),
    )

    assert isinstance(response, StreamingResponse)
