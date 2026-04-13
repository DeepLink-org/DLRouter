"""Request-id helpers for vLLM two-stage PD."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from dlrouter.core.service_discovery.base import NodeInfo


def build_encoded_request_id(prefill_info: NodeInfo, decode_info: NodeInfo) -> str:
    """Build a vLLM-style encoded request id for two-stage coordination."""
    prefill_addr = _get_coordination_address(prefill_info)
    decode_addr = _get_coordination_address(decode_info)
    suffix = uuid.uuid4().hex
    return f'___prefill_addr_{prefill_addr}___decode_addr_{decode_addr}_{suffix}'


def _get_coordination_address(node_info: NodeInfo) -> str:
    """Prefer ZMQ address for coordination metadata, fallback to HTTP."""
    return node_info.zmq_address or node_info.http_address
