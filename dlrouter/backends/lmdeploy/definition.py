"""LMDeploy backend definition."""

from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.lmdeploy.backend import LMDeployBackend
from dlrouter.constants import BackendType


LMDEPLOY_BACKEND_DEFINITION = BackendDefinition(
    backend_type=BackendType.LMDEPLOY,
    name='lmdeploy',
    backend_cls=LMDeployBackend,
)
