"""KV transfer adapters for vLLM PD execution."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from dlrouter.backends.vllm.request_id import build_encoded_request_id


if TYPE_CHECKING:
    from dlrouter.core.service_discovery.base import NodeInfo


class KVTransferAdapter(ABC):
    """Adapter for connector-specific KV transfer request shaping."""

    connector_name: str

    @abstractmethod
    def build_prefill_request(
        self,
        request_data: dict[str, Any],
        request_id: str,
        aborted_request_ids: list[str],
    ) -> dict[str, Any]:
        """Build a prefill-only request payload."""

    @abstractmethod
    def build_request_id(
        self,
        prefill_info: 'NodeInfo',
        decode_info: 'NodeInfo',
    ) -> str:
        """Build the connector-specific request id used across both stages."""

    @abstractmethod
    def extract_transfer_context(
        self,
        prefill_response_json: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Extract optional connector-specific transfer context from prefill response."""

    @abstractmethod
    def inject_decode_request(
        self,
        request_data: dict[str, Any],
        transfer_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the decode request payload using extracted transfer context."""

    def build_abort_payload(self, request_id: str) -> dict[str, Any]:
        """Build connector-specific abort metadata."""
        return {'aborted_request': [request_id]}


class VLLMKVTransferAdapter(KVTransferAdapter):
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
        prefill_info: 'NodeInfo',
        decode_info: 'NodeInfo',
    ) -> str:
        return build_encoded_request_id(prefill_info, decode_info)

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
