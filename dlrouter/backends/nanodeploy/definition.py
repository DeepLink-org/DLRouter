"""NanoDeploy backend definition."""

from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.nanodeploy.backend import NanoDeployBackend
from dlrouter.constants import BackendType


NANODEPLOY_BACKEND_DEFINITION = BackendDefinition(
    backend_type=BackendType.NANODEPLOY,
    name='nanodeploy',
    backend_cls=NanoDeployBackend,
)
