"""ZMQ-based heartbeat service discovery.

Implementation of HeartbeatServiceDiscovery using ZMQ ROUTER socket.
Suitable for vLLM P2P NCCL xPyD Router demo.

Message format (msgpack):
    {
        "type": "P" or "D",
        "http_address": "ip:port",
        "zmq_address": "ip:port"
    }
"""

import socket
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.core.service_discovery.heartbeat_discovery import (
    HeartbeatServiceDiscovery,
)
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery.zmq')


class ZMQHeartbeatDiscovery(HeartbeatServiceDiscovery):
    """ZMQ-based heartbeat service discovery for vLLM PD instances.

    Uses ZMQ ROUTER socket to listen for heartbeat messages from
    Prefill (P) and Decode (D) instances.
    """

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 30001,
        ping_timeout_seconds: int = 5,
        node_manager: Optional['NodeManager'] = None,
        models: Optional[list[str]] = None,
    ) -> None:
        """Initialize ZMQ heartbeat discovery.

        Args:
            host: Hostname to bind the ZMQ router socket.
            port: Port to bind the ZMQ router socket.
            ping_timeout_seconds: Timeout for instance expiration.
            node_manager: Optional NodeManager to sync discovered instances.
            models: List of model names served by vLLM instances.
        """
        if port == 0:
            raise ValueError('Port cannot be 0')

        super().__init__(
            node_manager=node_manager,
            models=models,
            ping_timeout_seconds=ping_timeout_seconds,
        )

        self._host = host or socket.gethostname()
        self._port = port

        # ZMQ resources (initialized lazily)
        self._context: Any = None
        self._router_socket: Any = None

    def _log_start(self) -> None:
        """Log ZMQ discovery start."""
        logger.info(f'ZMQ heartbeat discovery started on tcp://{self._host}:{self._port}')

    def _cleanup_resources(self) -> None:
        """Cleanup ZMQ resources."""
        if self._router_socket:
            self._router_socket.close()
            self._router_socket = None
        if self._context:
            self._context.term()
            self._context = None

    def _listen_loop(self) -> None:
        """Main listener loop for ZMQ heartbeat messages."""
        try:
            import zmq
        except ImportError:
            logger.error('pyzmq is required. Install: pip install pyzmq')
            self._running = False
            return

        try:
            import msgpack  # noqa: F401
        except ImportError:
            logger.error('msgpack is required. Install: pip install msgpack')
            self._running = False
            return

        self._context = zmq.Context()
        self._router_socket = self._context.socket(zmq.ROUTER)
        self._router_socket.bind(f'tcp://{self._host}:{self._port}')

        poller = zmq.Poller()
        poller.register(self._router_socket, zmq.POLLIN)

        while self._running:
            try:
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                if self._router_socket in socks:
                    self._handle_zmq_message()
            except Exception as e:
                if self._running:
                    logger.error(f'ZMQ listener error: {e}')

    def _handle_zmq_message(self) -> None:
        """Handle a single ZMQ heartbeat message."""
        import msgpack

        try:
            remote_address, message = self._router_socket.recv_multipart()
            data = msgpack.loads(message)

            http_address = data.get('http_address')
            zmq_address = data.get('zmq_address')
            instance_type = data.get('type')
            metadata = data.get('metadata')

            if not all([http_address, zmq_address, instance_type]):
                logger.warning(f'Invalid message from {remote_address}: {data}')
                return

            if instance_type == 'P':
                self._register_prefill(http_address, zmq_address, metadata=metadata)
            elif instance_type == 'D':
                self._register_decode(http_address, zmq_address, metadata=metadata)
            else:
                logger.warning(f'Unknown instance type from {remote_address}: {data}')
        except Exception as e:
            logger.error(f'Error handling ZMQ message: {e}')
