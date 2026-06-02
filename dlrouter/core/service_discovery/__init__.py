"""Service discovery module.

Provides unified service discovery abstraction for PD disaggregation.

Two discovery modes (based on research document analysis):
- STATIC: Manual configuration (SGLang mini_lb, vLLM disagg_proxy_demo, Mooncake, NIXL)
  Proxy gets P/D list from CLI args, does NOT read from registry.
- HEARTBEAT: Instance-initiated registration (vLLM P2P NCCL xPyD Router)
  P/D instances send heartbeat to Router with http+ZMQ addresses.

Usage:
    Production path uses backend.create_service_discovery(...) in app.py.
    Direct class instantiation is also supported:

    from dlrouter.core.service_discovery import (
        StaticServiceDiscovery,
        ZMQHeartbeatDiscovery,
    )
"""

from dlrouter.core.service_discovery.base import BaseServiceDiscovery, NodeInfo
from dlrouter.core.service_discovery.heartbeat_discovery import (
    HeartbeatServiceDiscovery,
)
from dlrouter.core.service_discovery.nanoctrl_discovery import (
    NanoCtrlServiceDiscovery,
)
from dlrouter.core.service_discovery.static_discovery import (
    StaticServiceDiscovery,
)
from dlrouter.core.service_discovery.zmq_heartbeat import (
    ZMQHeartbeatDiscovery,
)


# Export all classes
__all__ = [
    'BaseServiceDiscovery',
    'HeartbeatServiceDiscovery',
    'NanoCtrlServiceDiscovery',
    'NodeInfo',
    'StaticServiceDiscovery',
    'ZMQHeartbeatDiscovery',
]
