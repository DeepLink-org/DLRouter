"""Static service discovery implementation.

Manual configuration mode where nodes are added/removed via API or CLI.
Suitable for SGLang mini_lb, vLLM disagg_proxy_demo, vLLM-ascend proxy.
"""

from typing import TYPE_CHECKING, Optional

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery.base import BaseServiceDiscovery, NodeInfo
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery.static')


class StaticServiceDiscovery(BaseServiceDiscovery):
    """Static service discovery with initial node configuration.

    Initial prefill/decode instances are synced to NodeManager on start().

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
        self._initial_prefill = prefill_instances or []
        self._initial_decode = decode_instances or []

    # -- Lifecycle --

    def start(self) -> None:
        """Start static discovery — sync initial instances to NodeManager."""
        self._running = True
        for node in self._initial_prefill:
            self._sync_to_node_manager(node.http_address, EngineRole.PREFILL, node.zmq_address)
            logger.info(f'🔵 Add Prefill [HTTP:{node.http_address}, ZMQ:{node.zmq_address}]')
        for node in self._initial_decode:
            self._sync_to_node_manager(node.http_address, EngineRole.DECODE, node.zmq_address)
            logger.info(f'🔵 Add Decode [HTTP:{node.http_address}, ZMQ:{node.zmq_address}]')
        logger.info(
            f'Static service discovery started with {len(self._initial_prefill)}P, {len(self._initial_decode)}D'
        )

    def stop(self) -> None:
        """Stop static discovery."""
        self._running = False
        logger.info('Static service discovery stopped')
