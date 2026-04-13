"""Compatibility filtering and pair selection for vLLM PD."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from dlrouter.core.service_discovery.base import NodeInfo


class VLLMPairSelector:
    """Select compatible prefill/decode pairs for vLLM two-stage PD."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def filter_pairs(
        self,
        *,
        prefill_candidates: list[NodeInfo],
        decode_candidates: list[NodeInfo],
        model_name: str,
    ) -> list[tuple[NodeInfo, NodeInfo]]:
        """Return all compatible prefill/decode pairs."""
        return [
            (prefill, decode)
            for prefill in prefill_candidates
            for decode in decode_candidates
            if self._is_compatible(prefill, decode, model_name)
        ]

    def select_pair(
        self,
        *,
        prefill_candidates: list[NodeInfo],
        decode_candidates: list[NodeInfo],
        model_name: str,
    ) -> tuple[NodeInfo, NodeInfo] | None:
        """Select the next compatible pair using round-robin."""
        compatible_pairs = self.filter_pairs(
            prefill_candidates=prefill_candidates,
            decode_candidates=decode_candidates,
            model_name=model_name,
        )
        if not compatible_pairs:
            return None

        with self._lock:
            index = self._counter % len(compatible_pairs)
            self._counter += 1

        return compatible_pairs[index]

    def _is_compatible(
        self,
        prefill: NodeInfo,
        decode: NodeInfo,
        model_name: str,
    ) -> bool:
        if model_name not in prefill.models or model_name not in decode.models:
            return False

        prefill_metadata = prefill.metadata or {}
        decode_metadata = decode.metadata or {}

        for key in ('kv_connector', 'protocol_version'):
            if prefill_metadata.get(key) != decode_metadata.get(key):
                return False

        return self._endpoint_metadata(prefill_metadata) == self._endpoint_metadata(decode_metadata)

    def _endpoint_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        endpoint_metadata = metadata.get('endpoint_metadata', {})
        if isinstance(endpoint_metadata, dict):
            return endpoint_metadata
        return {}
