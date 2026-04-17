"""Tests for vLLM KV transfer adapters."""

import threading
from unittest.mock import MagicMock

from dlrouter.backends.vllm.kv_transfer import VLLMKVTransferAdapter
from dlrouter.constants import EngineRole
from dlrouter.models.node import NodeStatus


def test_vllm_build_prefill_request_forces_prefill_only_mode():
    adapter = VLLMKVTransferAdapter()

    request = adapter.build_prefill_request(
        {
            'model': 'qwen3-32b',
            'stream': True,
            'max_tokens': 64,
            'stream_options': {'include_usage': True},
        },
        request_id='req-1',
        aborted_request_ids=['req-old'],
    )

    assert request['stream'] is False
    assert request['max_tokens'] == 1
    assert request['min_tokens'] == 1
    assert request['kv_transfer_params']['do_remote_decode'] is True
    assert request['kv_transfer_params']['aborted_request'] == ['req-old']
    assert 'stream_options' not in request


def test_vllm_extract_transfer_context_returns_kv_transfer_params():
    adapter = VLLMKVTransferAdapter()
    payload = {'kv_transfer_params': {'remote_host': '10.0.0.9', 'remote_port': 20001}}

    assert adapter.extract_transfer_context(payload) == payload['kv_transfer_params']


def test_vllm_extract_transfer_context_returns_none_for_null_value():
    adapter = VLLMKVTransferAdapter()

    assert adapter.extract_transfer_context({'kv_transfer_params': None}) is None


def test_vllm_extract_transfer_context_returns_none_when_field_missing():
    adapter = VLLMKVTransferAdapter()

    assert adapter.extract_transfer_context({}) is None


def test_vllm_inject_decode_request_merges_transfer_context():
    adapter = VLLMKVTransferAdapter()

    request = adapter.inject_decode_request(
        {'model': 'qwen3-32b', 'messages': []},
        {'remote_host': '10.0.0.9', 'remote_port': 20001},
    )

    assert request['kv_transfer_params']['remote_host'] == '10.0.0.9'
    assert request['kv_transfer_params']['remote_port'] == 20001


def test_vllm_build_request_id_returns_encoded_request_id():
    adapter = VLLMKVTransferAdapter()
    nodes = {
        'http://10.0.0.1:13700': NodeStatus(
            role=EngineRole.PREFILL,
            models=['test-model'],
            zmq_address='10.0.0.1:30001',
        ),
        'http://10.0.0.2:13701': NodeStatus(
            role=EngineRole.DECODE,
            models=['test-model'],
            zmq_address='10.0.0.2:30002',
        ),
    }
    nm = MagicMock()
    nm.nodes = nodes
    nm._lock = threading.RLock()

    request_id = adapter.build_request_id(
        'http://10.0.0.1:13700',
        'http://10.0.0.2:13701',
        nm,
    )

    assert request_id.startswith('___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30002_')
