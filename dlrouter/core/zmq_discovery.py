"""ZMQ-based service discovery for vLLM PD disaggregation.

This module provides automatic discovery and registration of
vLLM Prefill and Decode instances through ZMQ messaging.

Based on the official vLLM disaggregated proxy implementation.
"""

import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from dlrouter.constants import EngineRole
from dlrouter.logger import get_logger

if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager

logger = get_logger('dlrouter.zmq_discovery')

# Default ping timeout in seconds
DEFAULT_PING_SECONDS = 5


class ZMQServiceDiscovery:
    """ZMQ-based service discovery for vLLM PD instances.

    Listens for heartbeat messages from Prefill (P) and Decode (D)
    instances and maintains a registry of active instances with
    automatic expiration.

    Message format (msgpack):
        {
            "type": "P" or "D",
            "http_address": "ip:port",
            "zmq_address": "ip:port"
        }
    """

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 30001,
        ping_timeout_seconds: int = DEFAULT_PING_SECONDS,
        node_manager: Optional['NodeManager'] = None,
        models: Optional[list[str]] = None,
    ) -> None:
        """Initialize ZMQ service discovery.

        Args:
            host: Hostname to bind the ZMQ router socket.
            port: Port to bind the ZMQ router socket.
            ping_timeout_seconds: Timeout in seconds for instance expiration.
            node_manager: Optional NodeManager to sync discovered instances.
            models: List of model names served by vLLM instances.
        """
        if port == 0:
            raise ValueError('Port cannot be 0')

        self._host = host or socket.gethostname()
        self._port = port
        self._ping_timeout = ping_timeout_seconds
        self._node_manager = node_manager
        self._models = models or []

        # Instance registries: http_address -> (zmq_address, expiration_timestamp)
        self._prefill_instances: dict[str, tuple[str, float]] = {}
        self._decode_instances: dict[str, tuple[str, float]] = {}

        # Thread-safe access to registries
        self._prefill_lock = threading.Lock()
        self._decode_lock = threading.Lock()

        # Counter for round-robin selection
        self._counter = 0
        self._counter_lock = threading.Lock()

        # Background listener thread
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False

        # ZMQ resources (initialized lazily)
        self._context = None
        self._router_socket = None

    def start(self) -> None:
        """Start the service discovery listener."""
        if self._running:
            return

        try:
            import zmq
        except ImportError:
            logger.error(
                'pyzmq is required for ZMQ service discovery. '
                'Install it with: pip install pyzmq'
            )
            raise

        try:
            import msgpack  # noqa: F401
        except ImportError:
            logger.error(
                'msgpack is required for ZMQ service discovery. '
                'Install it with: pip install msgpack'
            )
            raise

        self._context = zmq.Context()
        self._router_socket = self._context.socket(zmq.ROUTER)
        self._router_socket.bind(f'tcp://{self._host}:{self._port}')

        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name='zmq-discovery',
        )
        self._listener_thread.start()
        logger.info(
            f'ZMQ service discovery started on tcp://{self._host}:{self._port}'
        )

    def stop(self) -> None:
        """Stop the service discovery listener."""
        self._running = False
        if self._router_socket:
            self._router_socket.close()
            self._router_socket = None
        if self._context:
            self._context.term()
            self._context = None
        self._listener_thread = None
        logger.info('ZMQ service discovery stopped')

    def _listen_loop(self) -> None:
        """Main listener loop for ZMQ messages."""
        import msgpack
        import zmq

        poller = zmq.Poller()
        poller.register(self._router_socket, zmq.POLLIN)

        while self._running:
            try:
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                if self._router_socket in socks:
                    self._handle_message(msgpack)
            except Exception as e:
                if self._running:
                    logger.error(f'ZMQ listener error: {e}')

    def _handle_message(self, msgpack) -> None:
        """Handle a single incoming message."""
        try:
            remote_address, message = self._router_socket.recv_multipart()
            data = msgpack.loads(message)

            http_address = data.get('http_address')
            zmq_address = data.get('zmq_address')
            instance_type = data.get('type')

            if not all([http_address, zmq_address, instance_type]):
                logger.warning(f'Invalid message from {remote_address}: {data}')
                return

            expiration = time.time() + self._ping_timeout

            if instance_type == 'P':
                self._register_prefill(http_address, zmq_address, expiration)
            elif instance_type == 'D':
                self._register_decode(http_address, zmq_address, expiration)
            else:
                logger.warning(
                    f'Unknown instance type from {remote_address}: {data}'
                )
        except Exception as e:
            logger.error(f'Error handling ZMQ message: {e}')

    def _register_prefill(
        self,
        http_address: str,
        zmq_address: str,
        expiration: float,
    ) -> None:
        """Register a Prefill instance."""
        with self._prefill_lock:
            is_new = http_address not in self._prefill_instances
            self._prefill_instances[http_address] = (zmq_address, expiration)
            self._remove_expired(self._prefill_instances)

        if is_new:
            logger.info(f'🔵 Add Prefill [HTTP:{http_address}, ZMQ:{zmq_address}]')
            # Sync to node_manager if available
            self._sync_to_node_manager(http_address, EngineRole.PREFILL)

    def _register_decode(
        self,
        http_address: str,
        zmq_address: str,
        expiration: float,
    ) -> None:
        """Register a Decode instance."""
        with self._decode_lock:
            is_new = http_address not in self._decode_instances
            self._decode_instances[http_address] = (zmq_address, expiration)
            self._remove_expired(self._decode_instances)

        if is_new:
            logger.info(f'🔵 Add Decode [HTTP:{http_address}, ZMQ:{zmq_address}]')
            # Sync to node_manager if available
            self._sync_to_node_manager(http_address, EngineRole.DECODE)

    def _sync_to_node_manager(
        self,
        http_address: str,
        role: EngineRole,
    ) -> None:
        """Sync discovered instance to node_manager.

        Args:
            http_address: HTTP address of the instance.
            role: Engine role (PREFILL or DECODE).
        """
        if self._node_manager is None:
            return

        try:
            from dlrouter.models.node import NodeStatus

            # Ensure URL has http:// prefix
            node_url = (
                http_address
                if http_address.startswith('http')
                else f'http://{http_address}'
            )

            status = NodeStatus(
                role=role,
                models=self._models.copy(),
            )
            self._node_manager.add(node_url, status)
            logger.info(f'Synced {role.name} instance {node_url} to node_manager')
        except Exception as e:
            logger.error(f'Failed to sync instance to node_manager: {e}')

    def _remove_expired(self, instances: dict[str, tuple[str, float]]) -> None:
        """Remove expired instances from the registry.

        Iterates through instances in insertion order (Python 3.7+)
        and removes those that have expired.
        """
        now = time.time()
        expired = []
        for http_addr, (zmq_addr, exp_time) in instances.items():
            if exp_time <= now:
                expired.append((http_addr, zmq_addr))

        for http_addr, zmq_addr in expired:
            instances.pop(http_addr, None)
            logger.info(f'🔴 Remove expired [HTTP:{http_addr}, ZMQ:{zmq_addr}]')
            # Also remove from node_manager
            if self._node_manager is not None:
                try:
                    node_url = (
                        http_addr
                        if http_addr.startswith('http')
                        else f'http://{http_addr}'
                    )
                    self._node_manager.remove(node_url)
                    logger.info(f'Removed expired instance {node_url} from node_manager')
                except Exception as e:
                    logger.error(f'Failed to remove expired instance from node_manager: {e}')

    @property
    def prefill_instances(self) -> dict[str, tuple[str, float]]:
        """Get current prefill instances (copy)."""
        with self._prefill_lock:
            return dict(self._prefill_instances)

    @property
    def decode_instances(self) -> dict[str, tuple[str, float]]:
        """Get current decode instances (copy)."""
        with self._decode_lock:
            return dict(self._decode_instances)

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

    def select_prefill_instance(self) -> Optional[tuple[str, str]]:
        """Select a prefill instance using round-robin.

        Returns:
            Tuple of (http_address, zmq_address) or None if no instances.
        """
        with self._prefill_lock:
            self._remove_expired(self._prefill_instances)
            if not self._prefill_instances:
                return None

            items = list(self._prefill_instances.items())
            with self._counter_lock:
                idx = self._counter % len(items)
            http_addr, (zmq_addr, _) = items[idx]
            return http_addr, zmq_addr

    def select_decode_instance(self) -> Optional[tuple[str, str]]:
        """Select a decode instance using round-robin.

        Returns:
            Tuple of (http_address, zmq_address) or None if no instances.
        """
        with self._decode_lock:
            self._remove_expired(self._decode_instances)
            if not self._decode_instances:
                return None

            items = list(self._decode_instances.items())
            with self._counter_lock:
                idx = self._counter % len(items)
            http_addr, (zmq_addr, _) = items[idx]
            return http_addr, zmq_addr

    def select_pd_pair(self) -> Optional[tuple[tuple[str, str], tuple[str, str]]]:
        """Select a Prefill-Decode pair using round-robin.

        Returns:
            Tuple of ((p_http, p_zmq), (d_http, d_zmq)) or None.
        """
        prefill = self.select_prefill_instance()
        decode = self.select_decode_instance()

        if prefill is None or decode is None:
            return None

        # Increment counter after successful selection
        with self._counter_lock:
            self._counter += 1

        return prefill, decode

    def build_request_id(
        self,
        prefill_zmq_addr: str,
        decode_zmq_addr: str,
        base_id: str,
    ) -> str:
        """Build a request ID encoding PD addresses.

        The vLLM disaggregated serving uses request_id to pass
        the P and D ZMQ addresses for KV cache transfer coordination.

        Format: ___prefill_addr_{p_zmq}___decode_addr_{d_zmq}_{uuid}

        Args:
            prefill_zmq_addr: ZMQ address of prefill instance.
            decode_zmq_addr: ZMQ address of decode instance.
            base_id: Base UUID for the request.

        Returns:
            Encoded request ID string.
        """
        return (
            f'___prefill_addr_{prefill_zmq_addr}'
            f'___decode_addr_{decode_zmq_addr}_{base_id}'
        )

    def get_status(self) -> dict[str, Any]:
        """Get current status of the service discovery.

        Returns:
            Dict with prefill_count, decode_count, and running state.
        """
        return {
            'running': self._running,
            'prefill_count': self.get_prefill_count(),
            'decode_count': self.get_decode_count(),
            'prefill_instances': [
                {'http': k, 'zmq': v[0]}
                for k, v in self.prefill_instances.items()
            ],
            'decode_instances': [
                {'http': k, 'zmq': v[0]}
                for k, v in self.decode_instances.items()
            ],
        }
