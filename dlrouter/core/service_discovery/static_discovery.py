"""Static service discovery implementation.

Manual configuration mode where nodes are added/removed via API or CLI.
Suitable for SGLang mini_lb, vLLM disagg_proxy_demo, vLLM-ascend proxy.
"""

import threading
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery.base import BaseServiceDiscovery, NodeInfo
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery.static')


class StaticServiceDiscovery(BaseServiceDiscovery):
    """Static service discovery with manual node management.

    Nodes are manually registered via add_prefill/add_decode methods.
    No automatic discovery or heartbeat mechanism.

    Suitable for:
    - SGLang mini_lb (--prefill/--decode static list)
    - vLLM disagg_proxy_demo (--prefill/--decode static list)
    - vLLM-ascend MooncakeLayerwiseConnector (manual registration)
    - LMDeploy PD mode
    """

    def __init__(
        self,
        node_manager: Optional['NodeManager'] = None,
        models: Optional[list[str]] = None,
        prefill_instances: Optional[list[NodeInfo]] = None,
        decode_instances: Optional[list[NodeInfo]] = None,
    ) -> None:
        """Initialize static service discovery.

        Args:
            node_manager: Optional NodeManager to sync discovered instances.
            models: List of model names for node registration.
            prefill_instances: Initial prefill instances list.
            decode_instances: Initial decode instances list.
        """
        super().__init__(node_manager, models)

        # Counter for round-robin selection
        self._counter = 0
        self._counter_lock = threading.Lock()

        # Initialize with provided instances
        if prefill_instances:
            for node in prefill_instances:
                self._registry.upsert(node)

        if decode_instances:
            for node in decode_instances:
                self._registry.upsert(node)

    # -- Lifecycle --

    def start(self) -> None:
        """Start static discovery (no background task needed)."""
        self._running = True
        # Sync initial instances to NodeManager only after models are ready.
        for node in list(self.get_prefill_instances()):
            prepared = self._prepare_node_for_registration(node)
            if prepared is None:
                self._registry.remove(node)
                logger.warning(f'⚠️ Skip Prefill [HTTP:{node.http_address}] because models are unavailable')
                continue
            self._registry.upsert(prepared)
            self._sync_to_node_manager(prepared)
        for node in list(self.get_decode_instances()):
            prepared = self._prepare_node_for_registration(node)
            if prepared is None:
                self._registry.remove(node)
                logger.warning(f'⚠️ Skip Decode [HTTP:{node.http_address}] because models are unavailable')
                continue
            self._registry.upsert(prepared)
            self._sync_to_node_manager(prepared)
        logger.info(f'Static service discovery started with {self.get_prefill_count()}P, {self.get_decode_count()}D')

    def stop(self) -> None:
        """Stop static discovery."""
        self._running = False
        logger.info('Static service discovery stopped')

    # -- Node Selection --

    def select_prefill(self) -> Optional[NodeInfo]:
        """Select a prefill instance using round-robin."""
        instances = self.get_prefill_instances()
        if not instances:
            return None

        with self._counter_lock:
            idx = self._counter % len(instances)
            self._counter += 1

        return instances[idx]

    def select_decode(self) -> Optional[NodeInfo]:
        """Select a decode instance using round-robin."""
        instances = self.get_decode_instances()
        if not instances:
            return None

        with self._counter_lock:
            idx = self._counter % len(instances)

        return instances[idx]

    # -- Manual Node Management --

    def add_prefill(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        models: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Manually add a prefill instance.

        Args:
            http_address: HTTP address for API requests.
            zmq_address: ZMQ address for KV transfer (optional).
            models: List of models served.
            metadata: Additional metadata.
        """
        node = NodeInfo(
            http_address=http_address,
            zmq_address=zmq_address,
            role=EngineRole.PREFILL,
            models=models or self._models.copy(),
            metadata=metadata or {},
        )

        existing = next((item for item in self.get_prefill_instances() if item.http_address == http_address), None)
        is_new = existing is None
        prepared = self._prepare_node_for_registration(node)
        if prepared is None:
            logger.warning(f'⚠️ Skip Prefill [HTTP:{http_address}] because models are unavailable')
            return
        self._registry.upsert(prepared)

        if is_new:
            logger.info(f'🔵 Add Prefill [HTTP:{http_address}, ZMQ:{zmq_address}]')
            self._sync_to_node_manager(prepared)

    def add_decode(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        models: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Manually add a decode instance."""
        node = NodeInfo(
            http_address=http_address,
            zmq_address=zmq_address,
            role=EngineRole.DECODE,
            models=models or self._models.copy(),
            metadata=metadata or {},
        )

        existing = next((item for item in self.get_decode_instances() if item.http_address == http_address), None)
        is_new = existing is None
        prepared = self._prepare_node_for_registration(node)
        if prepared is None:
            logger.warning(f'⚠️ Skip Decode [HTTP:{http_address}] because models are unavailable')
            return
        self._registry.upsert(prepared)

        if is_new:
            logger.info(f'🔵 Add Decode [HTTP:{http_address}, ZMQ:{zmq_address}]')
            self._sync_to_node_manager(prepared)

    def remove_prefill(self, http_address: str) -> None:
        """Remove a prefill instance."""
        node = next((item for item in self.get_prefill_instances() if item.http_address == http_address), None)
        if node:
            self._registry.remove(node)

        if node:
            logger.info(f'🔴 Remove Prefill [HTTP:{http_address}]')
            self._remove_from_node_manager(node)

    def remove_decode(self, http_address: str) -> None:
        """Remove a decode instance."""
        node = next((item for item in self.get_decode_instances() if item.http_address == http_address), None)
        if node:
            self._registry.remove(node)

        if node:
            logger.info(f'🔴 Remove Decode [HTTP:{http_address}]')
            self._remove_from_node_manager(node)

    # -- Bulk Operations --

    def set_prefill_instances(self, instances: list[NodeInfo]) -> None:
        """Replace all prefill instances.

        Args:
            instances: New list of prefill instances.
        """
        current = {node.http_address: node for node in self.get_prefill_instances()}
        new = {node.http_address: node for node in instances}

        for key in current.keys() - new.keys():
            self._registry.remove(current[key])
            self._remove_from_node_manager(current[key])

        for key, node in new.items():
            if key not in current:
                self._sync_to_node_manager(node)
            self._registry.upsert(node)

        logger.info(f'Set {len(instances)} prefill instances')

    def set_decode_instances(self, instances: list[NodeInfo]) -> None:
        """Replace all decode instances."""
        current = {node.http_address: node for node in self.get_decode_instances()}
        new = {node.http_address: node for node in instances}

        for key in current.keys() - new.keys():
            self._registry.remove(current[key])
            self._remove_from_node_manager(current[key])

        for key, node in new.items():
            if key not in current:
                self._sync_to_node_manager(node)
            self._registry.upsert(node)

        logger.info(f'Set {len(instances)} decode instances')

    # -- Convenience Methods for CLI/API --

    def add_prefill_from_url(self, url: str, zmq_url: Optional[str] = None) -> None:
        """Add prefill from URL string.

        Args:
            url: HTTP URL (e.g., "http://10.0.0.1:8000" or "10.0.0.1:8000").
            zmq_url: Optional ZMQ URL for KV transfer.
        """
        http_addr = url.replace('http://', '').replace('https://', '')
        self.add_prefill(http_addr, zmq_url)

    def add_decode_from_url(self, url: str, zmq_url: Optional[str] = None) -> None:
        """Add decode from URL string."""
        http_addr = url.replace('http://', '').replace('https://', '')
        self.add_decode(http_addr, zmq_url)
