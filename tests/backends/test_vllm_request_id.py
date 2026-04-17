"""Tests for encoded vLLM request IDs."""

import threading
from unittest.mock import MagicMock

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


def test_build_encoded_request_id_prefers_zmq_addresses() -> None:
    from dlrouter.backends.vllm.request_id import build_encoded_request_id

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
    from dlrouter.backends.vllm.request_id import build_encoded_request_id

    nm = _make_node_manager(prefill_zmq=None, decode_zmq=None)

    request_id = build_encoded_request_id(
        'http://10.0.0.1:13700',
        'http://10.0.0.2:13701',
        nm,
    )

    assert request_id.startswith('___prefill_addr_10.0.0.1:13700___decode_addr_10.0.0.2:13701_')


def test_build_encoded_request_id_appends_unique_uuid_suffix() -> None:
    from dlrouter.backends.vllm.request_id import build_encoded_request_id

    nm = _make_node_manager(
        prefill_zmq='10.0.0.1:30001',
        decode_zmq='10.0.0.2:30002',
    )

    first = build_encoded_request_id('http://10.0.0.1:13700', 'http://10.0.0.2:13701', nm)
    second = build_encoded_request_id('http://10.0.0.1:13700', 'http://10.0.0.2:13701', nm)

    assert first != second
    assert first.rsplit('_', 1)[0] == second.rsplit('_', 1)[0]
