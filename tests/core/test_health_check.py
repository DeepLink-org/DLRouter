"""Tests for HealthChecker cleanup behavior."""

import threading
from unittest.mock import AsyncMock, MagicMock

from dlrouter.constants import EngineRole
from dlrouter.core.health_check import HealthChecker
from dlrouter.models.node import NodeStatus


class TestHealthChecker:
    def test_removes_stale_node_from_node_manager(self):
        node_manager = MagicMock()
        backend = MagicMock()
        backend.check_health = AsyncMock(return_value=False)
        node_manager.backend = backend
        node_manager.nodes = {
            'http://10.0.0.1:8000': NodeStatus(role=EngineRole.PREFILL),
        }

        checker = HealthChecker(
            node_manager,
            max_failures=1,
        )

        checker._check()

        node_manager.remove.assert_called_once_with('http://10.0.0.1:8000')


class TestLazyModelDiscovery:
    """Tests for _try_fetch_models — lazy model loading on healthy nodes."""

    def _make_manager(self, nodes: dict[str, NodeStatus]) -> MagicMock:
        node_manager = MagicMock()
        node_manager.nodes = nodes
        node_manager._lock = threading.RLock()
        node_manager._save_config = MagicMock()
        node_manager.backend = MagicMock()
        node_manager.backend.check_health = AsyncMock(return_value=True)
        return node_manager

    def test_fetches_models_for_healthy_node_with_empty_models(self):
        """When a node is healthy but models=[], fetch and fill models."""
        status = NodeStatus(role=EngineRole.PREFILL, models=[])
        node_manager = self._make_manager({'http://n1:8000': status})
        node_manager.backend.fetch_models.return_value = ['qwen3-32b']

        checker = HealthChecker(node_manager, max_failures=3)
        checker._try_fetch_models('http://n1:8000')

        node_manager.backend.fetch_models.assert_called_once_with('http://n1:8000')
        assert status.models == ['qwen3-32b']
        node_manager._save_config.assert_called_once()

    def test_skips_fetch_when_models_already_present(self):
        """When models are already populated, do nothing."""
        status = NodeStatus(role=EngineRole.PREFILL, models=['qwen3-32b'])
        node_manager = self._make_manager({'http://n1:8000': status})

        checker = HealthChecker(node_manager, max_failures=3)
        checker._try_fetch_models('http://n1:8000')

        node_manager.backend.fetch_models.assert_not_called()

    def test_skips_fetch_when_node_not_found(self):
        """When node_url not in nodes, do nothing."""
        node_manager = self._make_manager({})

        checker = HealthChecker(node_manager, max_failures=3)
        checker._try_fetch_models('http://n1:8000')

        node_manager.backend.fetch_models.assert_not_called()

    def test_no_update_when_fetch_returns_empty(self):
        """When fetch_models returns [], don't update."""
        status = NodeStatus(role=EngineRole.PREFILL, models=[])
        node_manager = self._make_manager({'http://n1:8000': status})
        node_manager.backend.fetch_models.return_value = []

        checker = HealthChecker(node_manager, max_failures=3)
        checker._try_fetch_models('http://n1:8000')

        assert status.models == []
        node_manager._save_config.assert_not_called()

    def test_handles_fetch_exception_gracefully(self):
        """When fetch_models raises, log warning and continue."""
        status = NodeStatus(role=EngineRole.PREFILL, models=[])
        node_manager = self._make_manager({'http://n1:8000': status})
        node_manager.backend.fetch_models.side_effect = ConnectionError('refused')

        checker = HealthChecker(node_manager, max_failures=3)
        checker._try_fetch_models('http://n1:8000')

        assert status.models == []
        node_manager._save_config.assert_not_called()

    def test_health_check_triggers_lazy_fetch(self):
        """Full integration: _check() calls _try_fetch_models for healthy nodes."""
        status = NodeStatus(role=EngineRole.PREFILL, models=[])
        node_manager = self._make_manager({'http://n1:8000': status})
        node_manager.backend.fetch_models.return_value = ['qwen3-32b']

        checker = HealthChecker(node_manager, max_failures=3)
        checker._check()

        node_manager.backend.fetch_models.assert_called_once_with('http://n1:8000')
        assert status.models == ['qwen3-32b']
