"""Backend adapters for DLRouter."""

from dlrouter.backends.base import BaseBackend
from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.factory import create_backend, get_backend_definition
from dlrouter.backends.lmdeploy import (
    LMDEPLOY_BACKEND_DEFINITION,
    LMDeployBackend,
    LMDeployPDConfig,
)
from dlrouter.backends.sglang import (
    SGLANG_BACKEND_DEFINITION,
    SGLangBackend,
    SGLangPDConfig,
)
from dlrouter.backends.vllm import (
    VLLM_BACKEND_DEFINITION,
    VLLMBackend,
    VLLMPDConfig,
)


__all__ = [
    'LMDEPLOY_BACKEND_DEFINITION',
    'SGLANG_BACKEND_DEFINITION',
    'VLLM_BACKEND_DEFINITION',
    'BackendDefinition',
    'BaseBackend',
    'LMDeployBackend',
    'LMDeployPDConfig',
    'SGLangBackend',
    'SGLangPDConfig',
    'VLLMBackend',
    'VLLMPDConfig',
    'create_backend',
    'get_backend_definition',
]
