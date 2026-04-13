"""Base service discovery interface.

This module provides the abstract base class for service discovery,
supporting two modes based on research document analysis:
- STATIC: Manual configuration (SGLang mini_lb, vLLM disagg_proxy_demo)
- HEARTBEAT: Instance registration + heartbeat (vLLM P2P NCCL xPyD Router)
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import EngineRole
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery')


@dataclass
class NodeInfo:
    """Information about a discovered node.

    Attributes:
        http_address: HTTP address for API requests (e.g., "10.0.0.1:8000").
        zmq_address: ZMQ address for KV transfer coordination (optional).
        role: Engine role (PREFILL or DECODE).
        models: List of models served by this node.
        metadata: Additional backend-specific metadata.
        expiration: Expiration timestamp for heartbeat mode (optional).
    """

    http_address: str
    zmq_address: Optional[str] = None
    role: EngineRole = EngineRole.HYBRID
    models: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    expiration: Optional[float] = None  # 心跳过期时间

    def is_expired(self) -> bool:
        """Check if this node info has expired (for heartbeat mode)."""
        if self.expiration is None:
            return False
        return self.expiration <= time.time()

    def to_http_url(self) -> str:
        """Convert to full HTTP URL."""
        if self.http_address.startswith('http'):
            return self.http_address
        return f'http://{self.http_address}'


class BaseServiceDiscovery(ABC):
    """Abstract base class for service discovery.

    Provides a unified interface for discovering Prefill and Decode
    instances across different backends (vLLM, SGLang, LMDeploy).

    Two discovery modes:
    - STATIC: Manual configuration, no automatic discovery
    - HEARTBEAT: Instances主动register via heartbeat
    """

    def __init__(
        self,
        node_manager: Optional['NodeManager'] = None,
        models: Optional[list[str]] = None,
    ) -> None:
        """Initialize service discovery.

        Args:
            node_manager: Optional NodeManager to sync discovered instances.
            models: List of model names for node registration.
        """
        self._node_manager = node_manager
        self._models = models or []
        self._running = False

    # -- Lifecycle --

    @abstractmethod
    def start(self) -> None:
        """Start the service discovery mechanism.

        For HEARTBEAT mode: starts listening for heartbeat messages.
        For STATIC mode: no background task needed.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the service discovery mechanism."""

    @property
    def running(self) -> bool:
        """Whether the discovery mechanism is running."""
        return self._running

    # -- Node Selection --

    @abstractmethod
    def select_prefill(self) -> Optional[NodeInfo]:
        """Select a prefill instance.

        Returns:
            NodeInfo for selected prefill instance, or None.
        """

    @abstractmethod
    def select_decode(self) -> Optional[NodeInfo]:
        """Select a decode instance.

        Returns:
            NodeInfo for selected decode instance, or None.
        """

    def select_pd_pair(self) -> Optional[tuple[NodeInfo, NodeInfo]]:
        """Select a Prefill-Decode pair.

        Default implementation: select P and D independently.
        Subclasses can override for coordinated selection.

        Returns:
            Tuple of (prefill, decode) NodeInfo, or None.
        """
        prefill = self.select_prefill()
        decode = self.select_decode()
        if prefill is None or decode is None:
            return None
        return prefill, decode

    # -- Node Management (for STATIC mode) --

    def add_prefill(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        models: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Manually add a prefill instance (STATIC mode).

        Args:
            http_address: HTTP address for API requests.
            zmq_address: ZMQ address for KV transfer (optional).
            models: List of models served.
            metadata: Additional metadata.
        """
        raise NotImplementedError('This discovery mode does not support manual add')

    def add_decode(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        models: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Manually add a decode instance (STATIC mode)."""
        raise NotImplementedError('This discovery mode does not support manual add')

    def remove_prefill(self, http_address: str) -> None:
        """Remove a prefill instance."""
        raise NotImplementedError('This discovery mode does not support manual remove')

    def remove_decode(self, http_address: str) -> None:
        """Remove a decode instance."""
        raise NotImplementedError('This discovery mode does not support manual remove')

    def remove_node_url(self, node_url: str) -> None:
        """Remove a node from discovery using a full node URL."""
        http_address = node_url.replace('http://', '').replace('https://', '')
        for remove_fn in (self.remove_prefill, self.remove_decode):
            try:
                remove_fn(http_address)
            except NotImplementedError:
                continue

    # -- Query --

    @abstractmethod
    def get_prefill_count(self) -> int:
        """Get number of active prefill instances."""

    @abstractmethod
    def get_decode_count(self) -> int:
        """Get number of active decode instances."""

    @abstractmethod
    def get_prefill_instances(self) -> list[NodeInfo]:
        """Get list of all prefill instances."""

    @abstractmethod
    def get_decode_instances(self) -> list[NodeInfo]:
        """Get list of all decode instances."""

    # -- NodeManager Sync --

    def _sync_to_node_manager(self, node_info: NodeInfo) -> None:
        """Sync discovered instance to NodeManager.

        Args:
            node_info: The discovered node information.
        """
        if self._node_manager is None:
            return

        try:
            from dlrouter.models.node import NodeStatus

            node_url = node_info.to_http_url()
            status = NodeStatus(
                role=node_info.role,
                models=node_info.models or self._models.copy(),
            )
            self._node_manager.add(node_url, status)
            logger.info(f'Synced {node_info.role.name} instance {node_url} to node_manager')
        except Exception as e:
            logger.error(f'Failed to sync instance to node_manager: {e}')

    def _remove_from_node_manager(self, node_info: NodeInfo) -> None:
        """Remove instance from NodeManager.

        Args:
            node_info: The node to remove.
        """
        if self._node_manager is None:
            return

        try:
            node_url = node_info.to_http_url()
            self._node_manager.remove(node_url)
            logger.info(f'Removed instance {node_url} from node_manager')
        except Exception as e:
            logger.error(f'Failed to remove instance from node_manager: {e}')

    # -- Status --

    def get_status(self) -> dict[str, Any]:
        """Get current status of the service discovery.

        Returns:
            Dict with running state, instance counts, and instance details.
        """
        return {
            'running': self._running,
            'prefill_count': self.get_prefill_count(),
            'decode_count': self.get_decode_count(),
            'prefill_instances': [{'http': n.http_address, 'zmq': n.zmq_address} for n in self.get_prefill_instances()],
            'decode_instances': [{'http': n.http_address, 'zmq': n.zmq_address} for n in self.get_decode_instances()],
        }

    # -- Request ID Building (for vLLM PD mode) --

    def build_request_id(
        self,
        prefill_info: NodeInfo,
        decode_info: NodeInfo,
        base_id: str,
    ) -> str:
        """Build a request ID encoding PD addresses.

        Default implementation for vLLM-style request_id encoding.
        Subclasses can override for backend-specific formats.

        Args:
            prefill_info: Prefill node information.
            decode_info: Decode node information.
            base_id: Base UUID for the request.

        Returns:
            Encoded request ID string.
        """
        # vLLM PD format: ___prefill_addr_{p_zmq}___decode_addr_{d_zmq}_{uuid}
        p_addr = prefill_info.zmq_address or prefill_info.http_address
        d_addr = decode_info.zmq_address or decode_info.http_address
        return f'___prefill_addr_{p_addr}___decode_addr_{d_addr}_{base_id}'
