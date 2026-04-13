"""Tests for HealthChecker cleanup behavior."""

from unittest.mock import AsyncMock, MagicMock

from dlrouter.constants import EngineRole
from dlrouter.core.health_check import HealthChecker
from dlrouter.models.node import NodeStatus


class TestHealthChecker:
    def test_removes_stale_node_from_service_discovery(self):
        node_manager = MagicMock()
        backend = MagicMock()
        backend.check_health = AsyncMock(return_value=False)
        node_manager.backend = backend
        node_manager.nodes = {
            'http://10.0.0.1:8000': NodeStatus(role=EngineRole.PREFILL),
        }

        service_discovery = MagicMock()

        checker = HealthChecker(
            node_manager,
            service_discovery=service_discovery,
            max_failures=1,
        )

        checker._check()

        node_manager.remove.assert_called_once_with('http://10.0.0.1:8000')
        service_discovery.remove_node_url.assert_called_once_with('http://10.0.0.1:8000')
