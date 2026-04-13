"""Base backend interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import ServiceDiscoveryMode
from dlrouter.models.node import NodeStatus


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager
    from dlrouter.core.service_discovery.base import BaseServiceDiscovery


@dataclass(frozen=True)
class PDRequestContext:
    """Context passed to backend-owned PD handlers."""

    node_manager: 'NodeManager'
    service_discovery: Optional['BaseServiceDiscovery'] = None
    request_key: Optional[str] = None


@dataclass
class CLIArg:
    """CLI argument definition for backend-specific parameters.

    Attributes:
        name: Argument name (without -- prefix).
        type: Python type (str, int, bool, etc.).
        default: Default value.
        help: Help text for the argument.
        choices: Optional list of valid choices.
    """

    name: str
    type: type
    default: Any
    help: str
    choices: Optional[list] = field(default=None)


class BaseBackend(ABC):
    """Abstract base class for inference backends.

    Each backend adapter knows how to communicate with
    a specific inference engine (lmdeploy, vllm, etc.).
    """

    @abstractmethod
    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward a request to the backend node.

        Args:
            node_url: The URL of the backend node.
            endpoint: API endpoint path.
            request_data: The request payload dict.
            stream: Whether to stream the response.

        Returns:
            Response text or async generator for stream.
        """

    @abstractmethod
    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream-forward request to backend node.

        Yields:
            Response chunks as bytes.
        """

    @abstractmethod
    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available model names from a node.

        Args:
            node_url: The URL of the backend node.

        Returns:
            List of available model names.
        """

    @abstractmethod
    async def check_health(self, node_url: str) -> bool:
        """Check health of a backend node.

        Args:
            node_url: The URL of the backend node.

        Returns:
            True if healthy, False otherwise.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close any persistent connections.

        Should be called during application shutdown.
        Subclasses should override this to cleanup resources.
        """

    def supports_pd_disagg(self) -> bool:
        """Whether this backend supports PD disagg."""
        return False

    async def handle_pd_request(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Handle request in PD disaggregation mode.

        Backends with their own PD implementation should override this.
        For example, vLLM uses ZMQ service discovery for P/D coordination.

        Args:
            request_data: The request payload.
            model_name: Requested model name.
            endpoint: API endpoint path.
            stream: Whether to stream response.
            context: Backend-owned PD execution context.

        Returns:
            Response (StreamingResponse or JSONResponse).

        Raises:
            NotImplementedError: If backend doesn't support PD.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement handle_pd_request')

    async def prefill_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Send a prefill-only request (PD disagg).

        Returns:
            Prefill result info dict, or None.
        """
        raise NotImplementedError('This backend does not support PD disagg.')

    async def decode_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        prefill_info: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Send a decode request with prefill info.

        Returns:
            Response text or async generator.
        """
        raise NotImplementedError('This backend does not support PD disagg.')

    @abstractmethod
    def deregister_node(self, node_url: str) -> None:
        """Cleanup when a node is removed.

        Called by NodeManager.remove() to let the backend
        release any resources tied to the node (e.g. PD
        connection pool entries).

        Args:
            node_url: URL of the node being removed.
        """

    def register_node(
        self,
        node_url: str,
        status: Optional[NodeStatus] = None,
    ) -> NodeStatus:
        """Register/discover a new node.

        Default implementation fetches models.

        Returns:
            NodeStatus with discovered info.
        """
        if status is None:
            status = NodeStatus()
        if not status.models:
            status.models = self.fetch_models(node_url)
        return status

    # -- CLI argument registration --

    @classmethod
    def get_cli_args(cls) -> list[CLIArg]:
        """Return backend-specific CLI arguments.

        Subclasses should override this to define their
        configuration parameters.

        Returns:
            List of CLIArg definitions.
        """
        return []

    @classmethod
    def parse_config(cls, **kwargs) -> dict[str, Any]:
        """Parse and validate backend-specific config from CLI args.

        Subclasses should override this to construct their
        configuration object from parsed CLI arguments.

        Args:
            **kwargs: CLI arguments as keyword arguments.

        Returns:
            Backend-specific configuration dict or object.
        """
        return {}

    @classmethod
    def create(cls, parsed_config: Any = None) -> 'BaseBackend':
        """Create a backend instance from parsed configuration."""
        return cls()

    def create_service_discovery(
        self,
        discovery_mode: ServiceDiscoveryMode,
        backend_config: dict[str, Any],
        node_manager: 'NodeManager',
    ) -> Optional['BaseServiceDiscovery']:
        """Create service discovery component for PD disaggregation.

        Backends that support PD disaggregation should override this method.
        The discovery_mode determines how Prefill/Decode instances are discovered:
        - STATIC: Manual configuration via CLI/API
        - HEARTBEAT: Instances send heartbeat messages

        Args:
            discovery_mode: The service discovery mode to use.
            backend_config: Backend-specific configuration dict.
            node_manager: The NodeManager instance.

        Returns:
            Service discovery instance, or None if not needed.
        """
        return None
