"""Backend factory."""

from typing import Any, Optional, Type

from dlrouter.backends.base import BaseBackend
from dlrouter.backends.lmdeploy_backend import LMDeployBackend
from dlrouter.backends.vllm_backend import VLLMBackend
from dlrouter.constants import BackendType


# Registry of backend classes
_BACKEND_REGISTRY: dict[BackendType, Type[BaseBackend]] = {
    BackendType.LMDEPLOY: LMDeployBackend,
    BackendType.VLLM: VLLMBackend,
}


def get_backend_class(backend_type: BackendType) -> Type[BaseBackend]:
    """Get the backend class for a given type.

    Args:
        backend_type: The backend type.

    Returns:
        The backend class.

    Raises:
        ValueError: If backend type is not supported.
    """
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(
            f'Unsupported backend: {backend_type}. '
            f'Available: {[e.value for e in BackendType]}',
        )
    return _BACKEND_REGISTRY[backend_type]


def create_backend(
    backend_type: BackendType,
    backend_config: Optional[dict[str, Any]] = None,
) -> BaseBackend:
    """Create a backend adapter instance.

    Args:
        backend_type: Backend type enum.
        backend_config: Backend-specific configuration dict.
            Will be parsed by the backend's parse_config() method.

    Returns:
        Backend adapter instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    backend_config = backend_config or {}
    backend_cls = get_backend_class(backend_type)
    parsed_config = backend_cls.parse_config(**backend_config)

    if backend_type == BackendType.LMDEPLOY:
        return LMDeployBackend(pd_config=parsed_config)
    if backend_type == BackendType.VLLM:
        # VLLMBackend doesn't take pd_config in constructor
        # (config is used by proxy_engine for ZMQ discovery)
        return VLLMBackend()

    # Should not reach here due to get_backend_class check
    raise ValueError(f'Unsupported backend: {backend_type}')
