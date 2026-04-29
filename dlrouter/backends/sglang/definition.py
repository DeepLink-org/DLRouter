"""SGLang backend definition."""

from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.sglang.backend import SGLangBackend
from dlrouter.constants import BackendType


SGLANG_BACKEND_DEFINITION = BackendDefinition(
    backend_type=BackendType.SGLANG,
    name='sglang',
    backend_cls=SGLangBackend,
)
