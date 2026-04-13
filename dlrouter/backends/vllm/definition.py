"""vLLM backend definition."""

from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.vllm.backend import VLLMBackend
from dlrouter.constants import BackendType


VLLM_BACKEND_DEFINITION = BackendDefinition(
    backend_type=BackendType.VLLM,
    name='vllm',
    backend_cls=VLLMBackend,
    capability_names=(
        'forward_request',
        'stream_forward',
        'fetch_models',
        'check_health',
        'register_node',
        'deregister_node',
    ),
)
