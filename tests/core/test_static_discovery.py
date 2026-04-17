"""Tests for StaticServiceDiscovery registration behavior."""

from unittest.mock import MagicMock

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo, StaticServiceDiscovery
from dlrouter.models.node import NodeStatus


class TestStaticDiscoveryRegistration:
    def test_start_syncs_initial_instances_to_node_manager(self):
        """Initial static nodes should be registered to NodeManager on start."""
        mock_node_manager = MagicMock()
        discovery = StaticServiceDiscovery(
            node_manager=mock_node_manager,
            models=['Qwen3-4B'],
            prefill_instances=[NodeInfo(http_address='10.0.0.1:8000', role=EngineRole.PREFILL)],
            decode_instances=[NodeInfo(http_address='10.0.0.2:8000', role=EngineRole.DECODE)],
        )

        discovery.start()

        assert mock_node_manager.add.call_count == 2
        # Verify first call is prefill
        first_call = mock_node_manager.add.call_args_list[0]
        assert first_call[0][0] == 'http://10.0.0.1:8000'
        assert isinstance(first_call[0][1], NodeStatus)
        assert first_call[0][1].role == EngineRole.PREFILL
        assert first_call[0][1].models == ['Qwen3-4B']
        # Verify second call is decode
        second_call = mock_node_manager.add.call_args_list[1]
        assert second_call[0][0] == 'http://10.0.0.2:8000'
        assert second_call[0][1].role == EngineRole.DECODE

    def test_start_without_node_manager_does_not_raise(self):
        """Start with no node_manager should not raise."""
        discovery = StaticServiceDiscovery(
            prefill_instances=[NodeInfo(http_address='10.0.0.1:8000', role=EngineRole.PREFILL)],
        )
        discovery.start()
        assert discovery.running is True
