"""Request-id helpers for vLLM two-stage PD."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


def build_encoded_request_id(
    prefill_url: str,
    decode_url: str,
    node_manager: NodeManager,
) -> str:
    """Build a vLLM-style encoded request id for two-stage coordination."""
    prefill_addr = _get_zmq_address(prefill_url, node_manager)
    decode_addr = _get_zmq_address(decode_url, node_manager)
    suffix = uuid.uuid4().hex
    return f'___prefill_addr_{prefill_addr}___decode_addr_{decode_addr}_{suffix}'


def _get_zmq_address(node_url: str, node_manager: NodeManager) -> str:
    """Get ZMQ address from NodeManager, fallback to stripping http:// from URL."""
    status = node_manager.nodes.get(node_url)
    if status and status.zmq_address:
        return status.zmq_address
    # Fallback: strip http:// prefix to get host:port
    return node_url.replace('http://', '').replace('https://', '')
