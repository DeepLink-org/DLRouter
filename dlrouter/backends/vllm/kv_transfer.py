"""KV transfer adapter for vLLM PD execution."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any


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
    return node_url.replace('http://', '').replace('https://', '')


class VLLMKVTransferAdapter:
    """Generic vLLM two-stage KV transfer adapter."""

    def _prepare_prefill_payload(self, request_data: dict[str, Any]) -> dict[str, Any]:
        payload = request_data.copy()
        payload['stream'] = False
        payload['max_tokens'] = 1
        payload['min_tokens'] = 1
        payload.pop('stream_options', None)
        if 'max_completion_tokens' in payload:
            payload['max_completion_tokens'] = 1
        return payload

    def build_request_id(
        self,
        prefill_url: str,
        decode_url: str,
        node_manager: NodeManager,
    ) -> str:
        return build_encoded_request_id(prefill_url, decode_url, node_manager)

    def build_prefill_request(
        self,
        request_data: dict[str, Any],
        request_id: str,
        aborted_request_ids: list[str],
    ) -> dict[str, Any]:
        payload = self._prepare_prefill_payload(request_data)
        payload['kv_transfer_params'] = {
            'do_remote_decode': True,
            'do_remote_prefill': False,
            'remote_engine_id': None,
            'remote_block_ids': None,
            'remote_host': None,
            'remote_port': None,
            'aborted_request': list(aborted_request_ids),
        }
        return payload

    def extract_transfer_context(
        self,
        prefill_response_json: dict[str, Any],
    ) -> dict[str, Any] | None:
        value = prefill_response_json.get('kv_transfer_params')
        if value is None:
            return None
        return dict(value)

    def inject_decode_request(
        self,
        request_data: dict[str, Any],
        transfer_context: dict[str, Any],
    ) -> dict[str, Any]:
        payload = request_data.copy()
        payload['kv_transfer_params'] = dict(transfer_context)
        return payload
