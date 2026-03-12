"""Backend factory."""

from dlrouter.backends.base import BaseBackend
from dlrouter.backends.lmdeploy_backend import (
    LMDeployBackend,
)
from dlrouter.backends.vllm_backend import VLLMBackend
from dlrouter.config import BackendConfig, LMDeployPDConfig
from dlrouter.constants import BackendType


def create_backend(
    config: BackendConfig,
    pd_config: LMDeployPDConfig = None,
) -> BaseBackend:
    """Create a backend adapter instance.

    Args:
        config: Backend configuration.
        pd_config: Optional PD disagg config.

    Returns:
        Backend adapter instance.

    Raises:
        ValueError: If backend type is not supported.
    """
    if config.type == BackendType.LMDEPLOY:
        return LMDeployBackend(pd_config=pd_config)
    if config.type == BackendType.VLLM:
        return VLLMBackend()
    raise ValueError(f'Unsupported backend: {config.type}. Available: {[e.value for e in BackendType]}')
