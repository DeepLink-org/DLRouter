"""App-level tests for backend-owned discovery-mode preference."""

from collections.abc import AsyncIterator
from typing import Any

from dlrouter.api import app as app_module
from dlrouter.backends.base import BaseBackend
from dlrouter.config import RouterConfig
from dlrouter.constants import BackendType, ServiceDiscoveryMode, ServingStrategy
from dlrouter.models.node import NodeStatus


class _BoundaryBackend(BaseBackend):
    """Backend test double that exposes discovery preference explicitly."""

    def __init__(self) -> None:
        self.preferred_configs: list[dict[str, Any]] = []
        self.created_discovery_mode: ServiceDiscoveryMode | None = None
        self.discovery = object()

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        raise NotImplementedError

    def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError

    def fetch_models(self, node_url: str) -> list[str]:
        return []

    async def check_health(self, node_url: str) -> bool:
        return True

    async def close(self) -> None:
        return None

    def deregister_node(self, node_url: str) -> None:
        return None

    def register_node(
        self,
        node_url: str,
        status: NodeStatus | None = None,
    ) -> NodeStatus:
        return status or NodeStatus()

    def preferred_discovery_mode(
        self,
        backend_config: dict[str, Any],
    ) -> ServiceDiscoveryMode:
        self.preferred_configs.append(dict(backend_config))
        return ServiceDiscoveryMode.STATIC

    def create_service_discovery(
        self,
        discovery_mode: ServiceDiscoveryMode,
        backend_config: dict[str, Any],
        node_manager: Any,
    ) -> object:
        self.created_discovery_mode = discovery_mode
        return self.discovery


def test_create_app_asks_backend_for_distserve_discovery_mode(monkeypatch):
    backend = _BoundaryBackend()
    backend_config = {'discovery_mode': 'heartbeat'}

    monkeypatch.setattr(
        app_module,
        'create_backend',
        lambda backend_type, config: backend,
    )

    app = app_module.create_app(
        RouterConfig(
            backend_type=BackendType.VLLM,
            serving_strategy=ServingStrategy.DISTSERVE,
            cache_status=False,
            backend_config=backend_config,
        )
    )

    assert backend.preferred_configs == [backend_config]
    assert backend.created_discovery_mode is ServiceDiscoveryMode.STATIC
    assert app.state.service_discovery is backend.discovery
