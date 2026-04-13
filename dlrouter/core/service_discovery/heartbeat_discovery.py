"""Heartbeat-based service discovery implementation.

Instance主动register via heartbeat messages.
Suitable for vLLM P2P NCCL xPyD Router demo.
"""

import threading
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery.base import BaseServiceDiscovery, NodeInfo
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery.heartbeat')

# Default ping timeout in seconds
DEFAULT_PING_TIMEOUT_SECONDS = 5


class HeartbeatServiceDiscovery(BaseServiceDiscovery):
    """Heartbeat-based service discovery.

    Instances actively register by sending heartbeat messages.
    Expired instances (no heartbeat within timeout) are automatically removed.

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
            ping_timeout_seconds: Timeout for instance expiration.
        """
        super().__init__(node_manager, models)

        self._ping_timeout = ping_timeout_seconds

        # Instance registries: http_address -> NodeInfo
        self._prefill_instances: dict[str, NodeInfo] = {}
        self._decode_instances: dict[str, NodeInfo] = {}

        # Thread-safe access
        self._prefill_lock = threading.Lock()
        self._decode_lock = threading.Lock()

        # Counter for round-robin selection
        self._counter = 0
        self._counter_lock = threading.Lock()

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
        """Main listener loop for heartbeat messages.

        Subclasses implement protocol-specific listening (ZMQ, HTTP, etc).
        """

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
            metadata: Additional metadata from heartbeat.
        """
        expiration = time.time() + self._ping_timeout
        node = NodeInfo(
            http_address=http_address,
            zmq_address=zmq_address,
            role=EngineRole.PREFILL,
            models=self._models.copy(),
            metadata=metadata or {},
            expiration=expiration,
        )

        with self._prefill_lock:
            is_new = http_address not in self._prefill_instances
            self._prefill_instances[http_address] = node
            self._remove_expired(self._prefill_instances)

        if is_new:
            logger.info(f'🔵 Add Prefill [HTTP:{http_address}, ZMQ:{zmq_address}]')
            self._sync_to_node_manager(node)

    def _register_decode(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a decode instance from heartbeat."""
        expiration = time.time() + self._ping_timeout
        node = NodeInfo(
            http_address=http_address,
            zmq_address=zmq_address,
            role=EngineRole.DECODE,
            models=self._models.copy(),
            metadata=metadata or {},
            expiration=expiration,
        )

        with self._decode_lock:
            is_new = http_address not in self._decode_instances
            self._decode_instances[http_address] = node
            self._remove_expired(self._decode_instances)

        if is_new:
            logger.info(f'🔵 Add Decode [HTTP:{http_address}, ZMQ:{zmq_address}]')
            self._sync_to_node_manager(node)

    def _remove_expired(self, instances: dict[str, NodeInfo]) -> None:
        """Remove expired instances from registry.

        Args:
            instances: The instance dict to clean (prefill or decode).
        """
        now = time.time()
        expired = []
        for http_addr, node in instances.items():
            if node.expiration and node.expiration <= now:
                expired.append((http_addr, node))

        for http_addr, node in expired:
            instances.pop(http_addr, None)
            logger.info(f'🔴 Remove expired [HTTP:{http_addr}, ZMQ:{node.zmq_address}]')
            self._remove_from_node_manager(node)

    def remove_prefill(self, http_address: str) -> None:
        """Remove a prefill instance explicitly."""
        with self._prefill_lock:
            node = self._prefill_instances.pop(http_address, None)

        if node:
            logger.info(f'🔴 Remove Prefill [HTTP:{http_address}]')
            self._remove_from_node_manager(node)

    def remove_decode(self, http_address: str) -> None:
        """Remove a decode instance explicitly."""
        with self._decode_lock:
            node = self._decode_instances.pop(http_address, None)

        if node:
            logger.info(f'🔴 Remove Decode [HTTP:{http_address}]')
            self._remove_from_node_manager(node)

    # -- Node Selection --

    def select_prefill(self) -> Optional[NodeInfo]:
        """Select a prefill instance using round-robin."""
        with self._prefill_lock:
            self._remove_expired(self._prefill_instances)
            instances = list(self._prefill_instances.values())
            if not instances:
                return None

            with self._counter_lock:
                idx = self._counter % len(instances)
                self._counter += 1

            return instances[idx]

    def select_decode(self) -> Optional[NodeInfo]:
        """Select a decode instance using round-robin."""
        with self._decode_lock:
            self._remove_expired(self._decode_instances)
            instances = list(self._decode_instances.values())
            if not instances:
                return None

            with self._counter_lock:
                idx = self._counter % len(instances)

            return instances[idx]

    # -- Query --

    def get_prefill_count(self) -> int:
        """Get number of active prefill instances."""
        with self._prefill_lock:
            self._remove_expired(self._prefill_instances)
            return len(self._prefill_instances)

    def get_decode_count(self) -> int:
        """Get number of active decode instances."""
        with self._decode_lock:
            self._remove_expired(self._decode_instances)
            return len(self._decode_instances)

    def get_prefill_instances(self) -> list[NodeInfo]:
        """Get list of all active prefill instances."""
        with self._prefill_lock:
            self._remove_expired(self._prefill_instances)
            return list(self._prefill_instances.values())

    def get_decode_instances(self) -> list[NodeInfo]:
        """Get list of all active decode instances."""
        with self._decode_lock:
            self._remove_expired(self._decode_instances)
            return list(self._decode_instances.values())
