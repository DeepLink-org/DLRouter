"""Tests for service discovery factory helpers."""

from unittest.mock import MagicMock

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import (
    NodeInfo,
    StaticServiceDiscovery,
)
from dlrouter.models.node import NodeStatus


class TestCreateServiceDiscoveryFactory:
    def test_static_discovery_syncs_initial_instances_to_node_manager(self):
        mock_nm = MagicMock()
        discovery = StaticServiceDiscovery(
            node_manager=mock_nm,
            models=['qwen3-32b'],
            prefill_instances=[
                NodeInfo(
                    http_address='10.0.0.1:8000',
                    role=EngineRole.PREFILL,
                    models=['qwen3-32b'],
                )
            ],
            decode_instances=[
                NodeInfo(
                    http_address='10.0.0.2:8000',
                    role=EngineRole.DECODE,
                    models=['qwen3-32b'],
                )
            ],
        )
        discovery.start()

        assert mock_nm.add.call_count == 2
        # Check that prefill and decode are registered
        prefill_status = mock_nm.add.call_args_list[0][0][1]
        decode_status = mock_nm.add.call_args_list[1][0][1]
        assert isinstance(prefill_status, NodeStatus)
        assert prefill_status.role == EngineRole.PREFILL
        assert isinstance(decode_status, NodeStatus)
        assert decode_status.role == EngineRole.DECODE
