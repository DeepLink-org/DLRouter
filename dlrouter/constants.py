"""Constants and enumerations for DLRouter."""

import enum
import os


# Heartbeat expiration in seconds
HEARTBEAT_EXPIRATION = int(os.getenv('DLROUTER_HEARTBEAT_EXPIRATION', '90'))

# Max latency samples to keep per node
LATENCY_DEQUE_LEN = 15

AIOHTTP_TIMEOUT = int(os.getenv('DLROUTER_AIOHTTP_TIMEOUT', '1800'))

# Health check timeout in seconds (increase for PD disaggregation scenarios
# where cache block GC may cause temporary unresponsiveness)
HEALTH_CHECK_TIMEOUT = int(os.getenv('DLROUTER_HEALTH_CHECK_TIMEOUT', '30'))

# Number of consecutive health check failures before removing a node
HEALTH_CHECK_MAX_FAILURES = int(os.getenv('DLROUTER_HEALTH_CHECK_MAX_FAILURES', '3'))


class RoutingStrategy(str, enum.Enum):
    """Supported routing strategies."""

    ROUND_ROBIN = 'round_robin'
    RANDOM = 'random'
    CONSISTENT_HASH = 'consistent_hash'
    MIN_EXPECTED_LATENCY = 'min_expected_latency'
    MIN_OBSERVED_LATENCY = 'min_observed_latency'
    PREFIX_CACHE = 'prefix_cache'


class BackendType(str, enum.Enum):
    """Supported inference backend types."""

    LMDEPLOY = 'lmdeploy'
    VLLM = 'vllm'
    SGLANG = 'sglang'


class ServingStrategy(str, enum.Enum):
    """Serving strategies."""

    HYBRID = 'hybrid'
    DISTSERVE = 'distserve'


class EngineRole(enum.Enum):
    """Engine role in PD disaggregation."""

    HYBRID = enum.auto()
    PREFILL = enum.auto()
    DECODE = enum.auto()


class ServiceDiscoveryMode(str, enum.Enum):
    """Service discovery modes for PD disaggregation.

    Based on research document analysis, only two modes exist:
    - STATIC: Manual configuration (SGLang mini_lb, vLLM disagg_proxy_demo, Mooncake, NIXL)
      Proxy gets P/D list from CLI args, does NOT read from registry.
    - HEARTBEAT: Instance-initiated registration (vLLM P2P NCCL xPyD Router)
      P/D instances send heartbeat to Router with http+ZMQ addresses.

    Note: Mooncake's etcd is for KV transfer layer (buffer discovery, handshake),
    NOT for Proxy to discover P/D nodes. Proxy still uses static config.
    """

    STATIC = 'static'  # 手动配置节点列表 (绝大多数场景)
    HEARTBEAT = 'heartbeat'  # 心跳注册模式 (仅 vLLM P2P NCCL)


class ErrorCode(enum.IntEnum):
    """Error codes."""

    MODEL_NOT_FOUND = 10400
    SERVICE_UNAVAILABLE = 10401
    API_TIMEOUT = 10402
    BACKEND_ERROR = 10403


ERROR_MESSAGES = {
    ErrorCode.MODEL_NOT_FOUND: ('The requested model does not exist.'),
    ErrorCode.SERVICE_UNAVAILABLE: ('Service is unavailable. Please retry later.'),
    ErrorCode.API_TIMEOUT: ('Failed to get response within timeout.'),
    ErrorCode.BACKEND_ERROR: ('Backend inference engine returned an error.'),
}
