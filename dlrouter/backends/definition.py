"""Backend definition metadata and helpers."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dlrouter.backends.base import BaseBackend, CLIArg
from dlrouter.constants import BackendType


BASE_BACKEND_CAPABILITIES = frozenset(
    {
        'check_health',
        'close',
        'create',
        'create_service_discovery',
        'deregister_node',
        'fetch_models',
        'forward_request',
        'get_cli_args',
        'parse_config',
        'register_node',
        'stream_forward',
    }
)


@dataclass(frozen=True)
class BackendDefinition:
    """Static definition for a backend type."""

    backend_type: BackendType
    name: str
    backend_cls: type[BaseBackend]
    capability_names: tuple[str, ...] = field(default_factory=tuple)
    create_fn: Optional[Callable[[Any], BaseBackend]] = None

    def get_cli_args(self) -> list[CLIArg]:
        """Return backend-specific CLI args."""
        return self.backend_cls.get_cli_args()

    def parse_config(self, **kwargs: Any) -> Any:
        """Parse backend-specific config."""
        return self.backend_cls.parse_config(**kwargs)

    def create_backend(
        self,
        raw_config: Optional[dict[str, Any]] = None,
    ) -> BaseBackend:
        """Create a backend instance from raw config."""
        raw_config = raw_config or {}
        parsed_config = self.parse_config(**raw_config)
        if self.create_fn is not None:
            return self.create_fn(parsed_config)
        return self.backend_cls.create(parsed_config)

    def supports(self, capability_name: str) -> bool:
        """Whether this backend exposes a named capability."""
        if capability_name in BASE_BACKEND_CAPABILITIES:
            return callable(getattr(self.backend_cls, capability_name, None))

        if capability_name in self.capability_names:
            return True

        backend_attr = getattr(self.backend_cls, capability_name, None)
        base_attr = getattr(BaseBackend, capability_name, None)
        return callable(backend_attr) and backend_attr is not base_attr
