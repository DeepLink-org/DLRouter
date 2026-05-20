"""Heartbeat-based service discovery implementation.

Instance主动register via heartbeat messages.
Suitable for vLLM P2P NCCL xPyD Router demo.
"""

import threading
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery.base import BaseServiceDiscovery
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery.heartbeat')

# Default ping timeout in seconds
DEFAULT_PING_TIMEOUT_SECONDS = 5


class HeartbeatServiceDiscovery(BaseServiceDiscovery):
    """Heartbeat-based service discovery.

    Instances actively register by sending heartbeat messages.
    Node liveness is managed by HealthChecker (not by heartbeat expiration).

    Suitable for:
    - vLLM P2P NCCL xPyD Router (ZMQ heartbeat)
    - Any backend with instance-initiated registration
    """

    def __init__(
        self,
        node_manager: Optional['NodeManager'] = None,
        models: Optional[list[str]] = None,
        ping_timeout_seconds: int = DEFAULT_PING_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize heartbeat service discovery.

        Args:
            node_manager: Optional NodeManager to sync discovered instances.
            models: List of model names for node registration.
            ping_timeout_seconds: Kept for API compatibility but not used for expiration.
        """
        super().__init__(node_manager, models)

        # Track registered addresses to detect new vs. existing heartbeats
        self._registered: set[str] = set()

        # Background listener thread
        self._listener_thread: Optional[threading.Thread] = None

    # -- Lifecycle --

    def start(self) -> None:
        """Start heartbeat listener."""
        if self._running:
            return

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name='heartbeat-discovery',
        )
        self._listener_thread.start()
        self._log_start()

    def stop(self) -> None:
        """Stop heartbeat listener."""
        self._running = False
        self._cleanup_resources()
        self._listener_thread = None
        logger.info('Heartbeat service discovery stopped')

    @abstractmethod
    def _listen_loop(self) -> None:
        """Main listener loop for heartbeat messages."""

    @abstractmethod
    def _cleanup_resources(self) -> None:
        """Cleanup protocol-specific resources (sockets, connections)."""

    @abstractmethod
    def _log_start(self) -> None:
        """Log discovery start with protocol-specific info."""

    # -- Instance Registration (called by _listen_loop) --

    def _register_prefill(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a prefill instance from heartbeat.

        Args:
            http_address: HTTP address from heartbeat.
            zmq_address: ZMQ address from heartbeat (optional).
            metadata: Additional metadata from heartbeat (unused, kept for API compat).
        """
        key = f'PREFILL:{http_address}'
        if key in self._registered:
            return  # Already registered, heartbeat renewal — no-op

        self._registered.add(key)
        logger.info(f'🔵 Add Prefill [HTTP:{http_address}, ZMQ:{zmq_address}]')
        self._sync_to_node_manager(http_address, EngineRole.PREFILL, zmq_address)

    def _register_decode(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a decode instance from heartbeat."""
        key = f'DECODE:{http_address}'
        if key in self._registered:
            return  # Already registered, heartbeat renewal — no-op

        self._registered.add(key)
        logger.info(f'🔵 Add Decode [HTTP:{http_address}, ZMQ:{zmq_address}]')
        self._sync_to_node_manager(http_address, EngineRole.DECODE, zmq_address)

    # -- Cache invalidation --

    def unregister_by_url(self, node_url: str) -> None:
        """Drop a node_url from the registered-address cache.

        Called by NodeManager when a node is removed (e.g. by HealthChecker
        after an instance crash). Without this, the cached key prevents the
        restarted instance from being re-added through subsequent heartbeats.

        Args:
            node_url: The full node URL stored in NodeManager
                (e.g. ``http://10.0.0.1:8000``).
        """
        address = node_url
        for prefix in ('http://', 'https://'):
            if address.startswith(prefix):
                address = address[len(prefix) :]
                break
        address = address.rstrip('/')

        for role_prefix in ('PREFILL', 'DECODE'):
            key = f'{role_prefix}:{address}'
            if key in self._registered:
                self._registered.discard(key)
                logger.info(
                    f'🟡 Unregister {role_prefix} [HTTP:{address}] from heartbeat cache; will accept re-registration',
                )
