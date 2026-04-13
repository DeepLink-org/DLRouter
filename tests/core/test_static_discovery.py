"""Tests for StaticServiceDiscovery registration behavior."""

from unittest.mock import MagicMock

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo, StaticServiceDiscovery


class TestStaticDiscoveryRegistration:
    def test_start_fetches_models_before_syncing_initial_instances(self):
        """Initial static nodes without models should fetch them before sync."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.side_effect = [['Qwen3-4B'], ['Qwen3-4B']]
        discovery = StaticServiceDiscovery(
            node_manager=mock_node_manager,
            prefill_instances=[NodeInfo(http_address='10.0.0.1:8000', role=EngineRole.PREFILL)],
            decode_instances=[NodeInfo(http_address='10.0.0.2:8000', role=EngineRole.DECODE)],
        )

        discovery.start()

        assert [node.models for node in discovery.get_prefill_instances()] == [['Qwen3-4B']]
        assert [node.models for node in discovery.get_decode_instances()] == [['Qwen3-4B']]
        assert mock_node_manager.backend.fetch_models.call_count == 2
        assert mock_node_manager.add.call_count == 2

    def test_start_skips_initial_instances_when_model_fetch_fails(self):
        """Initial static nodes should be skipped when models cannot be fetched."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.return_value = []
        discovery = StaticServiceDiscovery(
            node_manager=mock_node_manager,
            prefill_instances=[NodeInfo(http_address='10.0.0.1:8000', role=EngineRole.PREFILL)],
        )

        discovery.start()

        assert discovery.get_prefill_instances() == []
        mock_node_manager.add.assert_not_called()

    def test_add_prefill_fetches_models_before_sync(self):
        """Manual static registration should fetch models before sync."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.return_value = ['Qwen3-4B']
        discovery = StaticServiceDiscovery(node_manager=mock_node_manager)

        discovery.add_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        mock_node_manager.backend.fetch_models.assert_called_once_with('http://10.0.0.1:8000')
        instances = discovery.get_prefill_instances()
        assert len(instances) == 1
        assert instances[0].models == ['Qwen3-4B']
        mock_node_manager.add.assert_called_once()

    def test_add_decode_skips_registration_when_model_fetch_fails(self):
        """Manual static registration should skip nodes when model fetch fails."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.return_value = []
        discovery = StaticServiceDiscovery(node_manager=mock_node_manager)

        discovery.add_decode('10.0.0.2:8000', '10.0.0.2:30001')

        mock_node_manager.backend.fetch_models.assert_called_once_with('http://10.0.0.2:8000')
        assert discovery.get_decode_instances() == []
        mock_node_manager.add.assert_not_called()
