"""Service discovery module.

Provides unified service discovery abstraction for PD disaggregation.

Two discovery modes (based on research document analysis):
- STATIC: Manual configuration (SGLang mini_lb, vLLM disagg_proxy_demo, Mooncake, NIXL)
  Proxy gets P/D list from CLI args, does NOT read from registry.
- HEARTBEAT: Instance-initiated registration (vLLM P2P NCCL xPyD Router)
  P/D instances send heartbeat to Router with http+ZMQ addresses.

Note: Mooncake's etcd is for KV transfer layer (buffer discovery, handshake),
NOT for Proxy to discover P/D nodes. Proxy still uses static config.

Usage:
    from dlrouter.core.service_discovery import (
        ServiceDiscoveryMode,
        create_service_discovery,
        StaticServiceDiscovery,
        ZMQHeartbeatDiscovery,
    )

    # Create discovery based on mode
    discovery = create_service_discovery(
        mode=ServiceDiscoveryMode.STATIC,
        config={'prefill_urls': ['http://10.0.0.1:8000']},
        node_manager=manager,
    )
"""

from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import EngineRole, ServiceDiscoveryMode
from dlrouter.core.service_discovery.base import BaseServiceDiscovery, NodeInfo
from dlrouter.core.service_discovery.heartbeat_discovery import (
    HeartbeatServiceDiscovery,
)
from dlrouter.core.service_discovery.registry import ServiceDiscoveryRegistry
from dlrouter.core.service_discovery.static_discovery import (
    StaticServiceDiscovery,
)
from dlrouter.core.service_discovery.zmq_heartbeat import (
    ZMQHeartbeatDiscovery,
)


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


def create_service_discovery(
    mode: ServiceDiscoveryMode,
    config: dict[str, Any],
    node_manager: Optional['NodeManager'] = None,
) -> BaseServiceDiscovery:
    """Create service discovery instance based on mode.

    Args:
        mode: Service discovery mode (STATIC or HEARTBEAT).
        config: Mode-specific configuration dict.
        node_manager: Optional NodeManager to sync discovered instances.

    Returns:
        Service discovery instance.

    Config keys by mode:
        STATIC:
            - prefill_urls: list[str] prefill node URLs
            - decode_urls: list[str] decode node URLs
            - models: list[str] model names

        HEARTBEAT:
            - zmq_host: str ZMQ bind host
            - zmq_port: int ZMQ bind port
            - zmq_ping_timeout: int
            - models: list[str]
    """
    models = config.get('models', [])

    if mode == ServiceDiscoveryMode.STATIC:
        return _create_static_discovery(config, node_manager, models)

    if mode == ServiceDiscoveryMode.HEARTBEAT:
        return _create_heartbeat_discovery(config, node_manager, models)

    raise ValueError(f'Unknown service discovery mode: {mode}')


def _create_static_discovery(
    config: dict[str, Any],
    node_manager: Optional['NodeManager'],
    models: list[str],
) -> StaticServiceDiscovery:
    """Create static service discovery."""
    prefill_instances = []
    raw_prefill = config.get('prefill_urls', [])
    for item in raw_prefill:
        if isinstance(item, str):
            prefill_instances.append(
                NodeInfo(
                    http_address=_normalize_http_address(item),
                    role=EngineRole.PREFILL,
                    models=models,
                )
            )
        elif isinstance(item, NodeInfo):
            prefill_instances.append(item)

    decode_instances = []
    raw_decode = config.get('decode_urls', [])
    for item in raw_decode:
        if isinstance(item, str):
            decode_instances.append(
                NodeInfo(
                    http_address=_normalize_http_address(item),
                    role=EngineRole.DECODE,
                    models=models,
                )
            )
        elif isinstance(item, NodeInfo):
            decode_instances.append(item)

    return StaticServiceDiscovery(
        node_manager=node_manager,
        models=models,
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
    )


def _create_heartbeat_discovery(
    config: dict[str, Any],
    node_manager: Optional['NodeManager'],
    models: list[str],
) -> BaseServiceDiscovery:
    """Create heartbeat service discovery."""
    return ZMQHeartbeatDiscovery(
        host=config.get('zmq_host', '0.0.0.0'),
        port=config.get('zmq_port', 30001),
        ping_timeout_seconds=config.get('zmq_ping_timeout', 5),
        node_manager=node_manager,
        models=models,
    )


def _normalize_http_address(url: str) -> str:
    """Normalize configured node URL to host:port form."""
    return url.replace('http://', '').replace('https://', '')


# Export all classes
__all__ = [
    'BaseServiceDiscovery',
    'HeartbeatServiceDiscovery',
    'NodeInfo',
    'ServiceDiscoveryMode',
    'ServiceDiscoveryRegistry',
    'StaticServiceDiscovery',
    'ZMQHeartbeatDiscovery',
    'create_service_discovery',
]
