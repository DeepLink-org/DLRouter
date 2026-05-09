"""Tests for vLLM KV transfer adapters."""

import threading
from unittest.mock import MagicMock

import dlrouter.backends.vllm.kv_transfer as kv_transfer
from dlrouter.backends.vllm.kv_transfer import VLLMKVTransferAdapter, build_encoded_request_id
from dlrouter.constants import EngineRole
from dlrouter.models.node import NodeStatus


def _make_node_manager(prefill_zmq=None, decode_zmq=None):
    """Build a mock NodeManager with the given ZMQ addresses."""
    nodes = {
        'http://10.0.0.1:13700': NodeStatus(
            role=EngineRole.PREFILL,
            models=['test-model'],
            zmq_address=prefill_zmq,
        ),
        'http://10.0.0.2:13701': NodeStatus(
            role=EngineRole.DECODE,
            models=['test-model'],
            zmq_address=decode_zmq,
        ),
    }
    nm = MagicMock()
    nm.nodes = nodes
    nm._lock = threading.RLock()
    return nm


def test_vllm_kv_transfer_exposes_only_the_current_concrete_adapter():
    assert not hasattr(kv_transfer, 'KVTransferAdapter')
    assert not hasattr(VLLMKVTransferAdapter, 'build_abort_payload')


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
    nm = _make_node_manager(
        prefill_zmq='10.0.0.1:30001',
        decode_zmq='10.0.0.2:30002',
    )

    request_id = adapter.build_request_id(
        'http://10.0.0.1:13700',
        'http://10.0.0.2:13701',
        nm,
    )

    assert request_id.startswith('___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30002_')


def test_build_encoded_request_id_lives_with_kv_transfer_adapter() -> None:
    assert build_encoded_request_id.__module__ == 'dlrouter.backends.vllm.kv_transfer'


def test_build_encoded_request_id_prefers_zmq_addresses() -> None:
    nm = _make_node_manager(
        prefill_zmq='10.0.0.1:30001',
        decode_zmq='10.0.0.2:30002',
    )

    request_id = build_encoded_request_id(
        'http://10.0.0.1:13700',
        'http://10.0.0.2:13701',
        nm,
    )

    assert request_id.startswith('___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30002_')


def test_build_encoded_request_id_falls_back_to_http_addresses() -> None:
    nm = _make_node_manager(prefill_zmq=None, decode_zmq=None)

    request_id = build_encoded_request_id(
        'http://10.0.0.1:13700',
        'http://10.0.0.2:13701',
        nm,
    )

    assert request_id.startswith('___prefill_addr_10.0.0.1:13700___decode_addr_10.0.0.2:13701_')


def test_build_encoded_request_id_strips_dp_rank_from_fallback_addresses() -> None:
    nm = _make_node_manager(prefill_zmq=None, decode_zmq=None)
    nm.nodes = {
        'http://10.0.0.1:13700@2': NodeStatus(role=EngineRole.PREFILL, models=['test-model']),
        'http://10.0.0.2:13701@5': NodeStatus(role=EngineRole.DECODE, models=['test-model']),
    }

    request_id = build_encoded_request_id(
        'http://10.0.0.1:13700@2',
        'http://10.0.0.2:13701@5',
        nm,
    )

    assert '@' not in request_id
    assert request_id.startswith('___prefill_addr_10.0.0.1:13700___decode_addr_10.0.0.2:13701_')


def test_build_encoded_request_id_appends_unique_uuid_suffix() -> None:
    nm = _make_node_manager(
        prefill_zmq='10.0.0.1:30001',
        decode_zmq='10.0.0.2:30002',
    )

    first = build_encoded_request_id('http://10.0.0.1:13700', 'http://10.0.0.2:13701', nm)
    second = build_encoded_request_id('http://10.0.0.1:13700', 'http://10.0.0.2:13701', nm)

    assert first != second
    assert first.rsplit('_', 1)[0] == second.rsplit('_', 1)[0]
