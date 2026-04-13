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
DEFAULT_MODEL_FETCH_RETRY_SECONDS = 5


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
        self._model_fetch_retry_after: dict[str, float] = {}

        # Counter for round-robin selection
        self._counter = 0
        self._counter_lock = threading.Lock()

        # Background listener thread
        self._listener_thread: Optional[threading.Thread] = None

    def _prepare_new_node_for_registration(self, node: NodeInfo) -> Optional[NodeInfo]:
        """Throttle failed model fetches for not-yet-registered heartbeat nodes."""
        node_url = node.to_http_url()
        now = time.time()
        retry_after = self._model_fetch_retry_after.get(node_url, 0.0)
        if retry_after > now:
            return None

        prepared = self._prepare_node_for_registration(node)
        if prepared is None:
            self._model_fetch_retry_after[node_url] = now + DEFAULT_MODEL_FETCH_RETRY_SECONDS
            return None

        self._model_fetch_retry_after.pop(node_url, None)
        return prepared

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
        existing = next((item for item in self.get_prefill_instances() if item.http_address == http_address), None)
        is_new = existing is None
        expiration = time.time() + self._ping_timeout
        node = NodeInfo(
            http_address=http_address,
            zmq_address=zmq_address,
            role=EngineRole.PREFILL,
            models=existing.models.copy() if existing else self._models.copy(),
            metadata=metadata if metadata is not None else (existing.metadata.copy() if existing else {}),
            expiration=expiration,
        )

        if not is_new:
            self._registry.upsert(node)
            self._remove_expired()
            return

        prepared = self._prepare_new_node_for_registration(node)
        self._remove_expired()
        if prepared is None:
            return

        self._registry.upsert(prepared)
        logger.info(f'🔵 Add Prefill [HTTP:{http_address}, ZMQ:{zmq_address}]')
        self._sync_to_node_manager(prepared)

    def _register_decode(
        self,
        http_address: str,
        zmq_address: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a decode instance from heartbeat."""
        existing = next((item for item in self.get_decode_instances() if item.http_address == http_address), None)
        is_new = existing is None
        expiration = time.time() + self._ping_timeout
        node = NodeInfo(
            http_address=http_address,
            zmq_address=zmq_address,
            role=EngineRole.DECODE,
            models=existing.models.copy() if existing else self._models.copy(),
            metadata=metadata if metadata is not None else (existing.metadata.copy() if existing else {}),
            expiration=expiration,
        )

        if not is_new:
            self._registry.upsert(node)
            self._remove_expired()
            return

        prepared = self._prepare_new_node_for_registration(node)
        self._remove_expired()
        if prepared is None:
            return

        self._registry.upsert(prepared)
        logger.info(f'🔵 Add Decode [HTTP:{http_address}, ZMQ:{zmq_address}]')
        self._sync_to_node_manager(prepared)

    def _remove_expired(self) -> None:
        """Remove expired instances from the unified registry."""
        now = time.time()
        expired = [node for node in self._registry.get_all_instances() if node.expiration and node.expiration <= now]

        for node in expired:
            self._registry.remove(node)
            self._model_fetch_retry_after.pop(node.to_http_url(), None)
            logger.info(f'🔴 Remove expired [HTTP:{node.http_address}, ZMQ:{node.zmq_address}]')
            self._remove_from_node_manager(node)

    def remove_prefill(self, http_address: str) -> None:
        """Remove a prefill instance explicitly."""
        node = next((item for item in self.get_prefill_instances() if item.http_address == http_address), None)
        if node:
            self._registry.remove(node)
            self._model_fetch_retry_after.pop(node.to_http_url(), None)

        if node:
            logger.info(f'🔴 Remove Prefill [HTTP:{http_address}]')
            self._remove_from_node_manager(node)

    def remove_decode(self, http_address: str) -> None:
        """Remove a decode instance explicitly."""
        node = next((item for item in self.get_decode_instances() if item.http_address == http_address), None)
        if node:
            self._registry.remove(node)
            self._model_fetch_retry_after.pop(node.to_http_url(), None)

        if node:
            logger.info(f'🔴 Remove Decode [HTTP:{http_address}]')
            self._remove_from_node_manager(node)

    # -- Node Selection --

    def select_prefill(self) -> Optional[NodeInfo]:
        """Select a prefill instance using round-robin."""
        self._remove_expired()
        instances = self.get_prefill_instances()
        if not instances:
            return None

        with self._counter_lock:
            idx = self._counter % len(instances)
            self._counter += 1

        return instances[idx]

    def select_decode(self) -> Optional[NodeInfo]:
        """Select a decode instance using round-robin."""
        self._remove_expired()
        instances = self.get_decode_instances()
        if not instances:
            return None

        with self._counter_lock:
            idx = self._counter % len(instances)

        return instances[idx]

    # -- Query --

    def get_prefill_count(self) -> int:
        self._remove_expired()
        return super().get_prefill_count()

    def get_decode_count(self) -> int:
        self._remove_expired()
        return super().get_decode_count()

    def get_prefill_instances(self) -> list[NodeInfo]:
        self._remove_expired()
        return super().get_prefill_instances()

    def get_decode_instances(self) -> list[NodeInfo]:
        self._remove_expired()
        return super().get_decode_instances()
