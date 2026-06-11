"""DLEngine backend definition."""

from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.dlengine.backend import DLEngineBackend
from dlrouter.constants import BackendType


DLENGINE_BACKEND_DEFINITION = BackendDefinition(
    backend_type=BackendType.DLENGINE,
    name='dlengine',
    backend_cls=DLEngineBackend,
)
