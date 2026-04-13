"""Tests for encoded vLLM request IDs."""

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo


def test_build_encoded_request_id_prefers_zmq_addresses() -> None:
    from dlrouter.backends.vllm.request_id import build_encoded_request_id

    prefill = NodeInfo(
        http_address='10.0.0.1:13700',
        zmq_address='10.0.0.1:30001',
        role=EngineRole.PREFILL,
    )
    decode = NodeInfo(
        http_address='10.0.0.2:13701',
        zmq_address='10.0.0.2:30002',
        role=EngineRole.DECODE,
    )

    request_id = build_encoded_request_id(prefill, decode)

    assert request_id.startswith('___prefill_addr_10.0.0.1:30001___decode_addr_10.0.0.2:30002_')


def test_build_encoded_request_id_falls_back_to_http_addresses() -> None:
    from dlrouter.backends.vllm.request_id import build_encoded_request_id

    prefill = NodeInfo(
        http_address='10.0.0.1:13700',
        role=EngineRole.PREFILL,
    )
    decode = NodeInfo(
        http_address='10.0.0.2:13701',
        role=EngineRole.DECODE,
    )

    request_id = build_encoded_request_id(prefill, decode)

    assert request_id.startswith('___prefill_addr_10.0.0.1:13700___decode_addr_10.0.0.2:13701_')


def test_build_encoded_request_id_appends_unique_uuid_suffix() -> None:
    from dlrouter.backends.vllm.request_id import build_encoded_request_id

    prefill = NodeInfo(
        http_address='10.0.0.1:13700',
        zmq_address='10.0.0.1:30001',
        role=EngineRole.PREFILL,
    )
    decode = NodeInfo(
        http_address='10.0.0.2:13701',
        zmq_address='10.0.0.2:30002',
        role=EngineRole.DECODE,
    )

    first = build_encoded_request_id(prefill, decode)
    second = build_encoded_request_id(prefill, decode)

    assert first != second
    assert first.rsplit('_', 1)[0] == second.rsplit('_', 1)[0]
