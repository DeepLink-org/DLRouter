"""Backend factory."""

from typing import Any, Optional

from dlrouter.backends.base import BaseBackend
from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.lmdeploy import LMDEPLOY_BACKEND_DEFINITION
from dlrouter.backends.vllm import VLLM_BACKEND_DEFINITION
from dlrouter.constants import BackendType


# Registry of backend definitions
_BACKEND_REGISTRY: dict[BackendType, BackendDefinition] = {
    BackendType.LMDEPLOY: LMDEPLOY_BACKEND_DEFINITION,
    BackendType.VLLM: VLLM_BACKEND_DEFINITION,
}


def get_backend_definition(backend_type: BackendType) -> BackendDefinition:
    """Get the backend definition for a given type.

    Args:
        backend_type: The backend type.

    Returns:
        The backend definition.

    Raises:
        ValueError: If backend type is not supported.
    """
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(
            f'Unsupported backend: {backend_type}. Available: {[e.value for e in BackendType]}',
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
    definition = get_backend_definition(backend_type)
    return definition.create_backend(backend_config)
