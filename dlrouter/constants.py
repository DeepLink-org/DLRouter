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


class BackendType(str, enum.Enum):
    """Supported inference backend types."""

    LMDEPLOY = 'lmdeploy'
    VLLM = 'vllm'


class ServingStrategy(str, enum.Enum):
    """Serving strategies."""

    HYBRID = 'hybrid'
    DISTSERVE = 'distserve'


class EngineRole(enum.Enum):
    """Engine role in PD disaggregation."""

    HYBRID = enum.auto()
    PREFILL = enum.auto()
    DECODE = enum.auto()


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
