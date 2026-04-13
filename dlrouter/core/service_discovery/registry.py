"""Unified registry for discovered service instances."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from dlrouter.constants import EngineRole


if TYPE_CHECKING:
    from dlrouter.core.service_discovery.base import NodeInfo


class ServiceDiscoveryRegistry:
    """Thread-safe registry for discovered nodes."""

    def __init__(self) -> None:
        self._instances: dict[str, NodeInfo] = {}
        self._lock = threading.RLock()

    def upsert(self, node: NodeInfo) -> None:
        """Insert or replace a node record."""
        with self._lock:
            self._instances[self._get_key(node)] = node

    def remove(self, node_or_key: NodeInfo | str) -> NodeInfo | None:
        """Remove a node by object or key."""
        key = node_or_key if isinstance(node_or_key, str) else self._get_key(node_or_key)
        with self._lock:
            return self._instances.pop(key, None)

    def get_all_instances(self) -> list[NodeInfo]:
        """Return all registered instances."""
        with self._lock:
            return list(self._instances.values())

    def get_prefill_instances(self) -> list[NodeInfo]:
        """Return all prefill instances."""
        return self.filter_instances(role=EngineRole.PREFILL)

    def get_decode_instances(self) -> list[NodeInfo]:
        """Return all decode instances."""
        return self.filter_instances(role=EngineRole.DECODE)

    def filter_instances(
        self,
        *,
        role: EngineRole | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[NodeInfo]:
        """Return instances filtered by role, model, and metadata subset."""
        result = self.get_all_instances()

        if role is not None:
            result = [node for node in result if node.role is role]

        if model is not None:
            result = [node for node in result if model in node.models]

        if metadata:
            result = [
                node for node in result if all(node.metadata.get(key) == value for key, value in metadata.items())
            ]

        return result

    def _get_key(self, node: NodeInfo) -> str:
        return f'{node.role.value}:{node.http_address}'
