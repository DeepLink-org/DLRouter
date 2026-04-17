"""Base service discovery interface.

This module provides the abstract base class for service discovery,
supporting two modes based on research document analysis:
- STATIC: Manual configuration (SGLang mini_lb, vLLM disagg_proxy_demo)
- HEARTBEAT: Instance registration + heartbeat (vLLM P2P NCCL xPyD Router)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from dlrouter.constants import EngineRole
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery')


@dataclass
class NodeInfo:
    """Lightweight transport object for discovered node info.

    Used internally by service discovery to pass heartbeat/config data
    before registering into NodeManager.

    Attributes:
        http_address: HTTP address for API requests (e.g., "10.0.0.1:8000").
        zmq_address: ZMQ address for KV transfer coordination (optional).
        role: Engine role (PREFILL or DECODE).
        models: List of models served by this node.
    """

    http_address: str
    zmq_address: Optional[str] = None
    role: EngineRole = EngineRole.HYBRID
    models: list[str] = field(default_factory=list)


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
        """Start the service discovery mechanism."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the service discovery mechanism."""

    @property
    def running(self) -> bool:
        """Whether the discovery mechanism is running."""
        return self._running

    # -- NodeManager Sync --

    def _sync_to_node_manager(
        self,
        http_address: str,
        role: EngineRole,
        zmq_address: Optional[str] = None,
    ) -> None:
        """Register a discovered node into NodeManager.

        Args:
            http_address: HTTP address (e.g. "10.0.0.1:8000").
            role: The engine role (PREFILL or DECODE).
            zmq_address: Optional ZMQ address for KV transfer coordination.
        """
        if self._node_manager is None:
            return

        try:
            from dlrouter.models.node import NodeStatus

            node_url = f'http://{http_address}' if not http_address.startswith('http') else http_address
            status = NodeStatus(
                role=role,
                zmq_address=zmq_address,
                models=self._models.copy(),
            )
            self._node_manager.add(node_url, status)
            logger.info(f'Synced {role.name} instance {node_url} to node_manager')
        except Exception as e:
            logger.error(f'Failed to sync instance to node_manager: {e}')
