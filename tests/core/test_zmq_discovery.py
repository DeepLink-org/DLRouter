"""Tests for ZMQHeartbeatDiscovery."""

import time
from unittest.mock import MagicMock, patch

import pytest

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo, ZMQHeartbeatDiscovery
from dlrouter.models.node import NodeStatus


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_values(self):
        discovery = ZMQHeartbeatDiscovery()
        assert discovery._host == '0.0.0.0'
        assert discovery._port == 30001
        assert discovery._ping_timeout == 5
        assert discovery._running is False
        assert discovery._node_manager is None
        assert discovery._models == []

    def test_custom_values(self):
        discovery = ZMQHeartbeatDiscovery(
            host='127.0.0.1',
            port=40001,
            ping_timeout_seconds=10,
        )
        assert discovery._host == '127.0.0.1'
        assert discovery._port == 40001
        assert discovery._ping_timeout == 10

    def test_port_zero_raises(self):
        with pytest.raises(ValueError, match='Port cannot be 0'):
            ZMQHeartbeatDiscovery(port=0)

    def test_init_with_node_manager_and_models(self):
        """Test initialization with node_manager and models (new feature)."""
        mock_node_manager = MagicMock()
        models = ['kimi-k2.5', 'qwen-72b']

        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=models,
        )

        assert discovery._node_manager is mock_node_manager
        assert discovery._models == models


# ---------------------------------------------------------------------------
# Instance Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_prefill_instance(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        assert discovery.get_prefill_count() == 1
        instances = discovery.get_prefill_instances()
        assert len(instances) == 1
        assert instances[0].http_address == '10.0.0.1:8000'
        assert instances[0].zmq_address == '10.0.0.1:30001'

    def test_register_decode_instance(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        assert discovery.get_decode_count() == 1
        instances = discovery.get_decode_instances()
        assert len(instances) == 1
        assert instances[0].http_address == '10.0.0.2:8000'
        assert instances[0].zmq_address == '10.0.0.2:30001'

    def test_handle_zmq_message_preserves_optional_metadata(self):
        discovery = ZMQHeartbeatDiscovery()
        mock_socket = MagicMock()
        mock_socket.recv_multipart.return_value = [
            b'worker-a',
            __import__('msgpack').dumps(
                {
                    'type': 'P',
                    'http_address': '10.0.0.1:8000',
                    'zmq_address': '10.0.0.1:30001',
                    'metadata': {'kv_connector': 'mooncake', 'protocol_version': 'v1'},
                }
            ),
        ]
        discovery._router_socket = mock_socket

        discovery._handle_zmq_message()

        instances = discovery.get_prefill_instances()
        assert len(instances) == 1
        assert instances[0].metadata == {
            'kv_connector': 'mooncake',
            'protocol_version': 'v1',
        }

    def test_expired_instances_removed(self):
        discovery = ZMQHeartbeatDiscovery()
        past_time = time.time() - 10  # Already expired

        discovery._registry.upsert(
            NodeInfo(
                http_address='expired:8000',
                zmq_address='expired:30001',
                role=EngineRole.PREFILL,
                expiration=past_time,
            )
        )

        # get_prefill_count triggers cleanup
        assert discovery.get_prefill_count() == 0


# ---------------------------------------------------------------------------
# Instance Selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_select_prefill_instance_round_robin(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')
        discovery._register_prefill('10.0.0.2:8000', '10.0.0.2:30001')

        result = discovery.select_prefill()
        assert result is not None
        assert result.http_address in ['10.0.0.1:8000', '10.0.0.2:8000']

    def test_select_decode_instance_round_robin(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_decode('10.0.0.3:8000', '10.0.0.3:30001')
        discovery._register_decode('10.0.0.4:8000', '10.0.0.4:30001')

        result = discovery.select_decode()
        assert result is not None
        assert result.http_address in ['10.0.0.3:8000', '10.0.0.4:8000']

    def test_select_prefill_returns_none_when_empty(self):
        discovery = ZMQHeartbeatDiscovery()
        assert discovery.select_prefill() is None

    def test_select_decode_returns_none_when_empty(self):
        discovery = ZMQHeartbeatDiscovery()
        assert discovery.select_decode() is None

    def test_select_pd_pair(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        result = discovery.select_pd_pair()
        assert result is not None
        prefill, decode = result
        assert prefill.http_address == '10.0.0.1:8000'
        assert prefill.zmq_address == '10.0.0.1:30001'
        assert decode.http_address == '10.0.0.2:8000'
        assert decode.zmq_address == '10.0.0.2:30001'

    def test_select_pd_pair_returns_none_without_prefill(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        assert discovery.select_pd_pair() is None

    def test_select_pd_pair_returns_none_without_decode(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        assert discovery.select_pd_pair() is None


# ---------------------------------------------------------------------------
# Legacy request-id cleanup
# ---------------------------------------------------------------------------


class TestLegacyRequestIdCleanup:
    def test_build_request_id_is_no_longer_exposed(self):
        discovery = ZMQHeartbeatDiscovery()
        assert not hasattr(discovery, 'build_request_id')


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_get_status(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        status = discovery.get_status()
        assert status['running'] is False
        assert status['prefill_count'] == 1
        assert status['decode_count'] == 1
        assert len(status['prefill_instances']) == 1
        assert len(status['decode_instances']) == 1


# ---------------------------------------------------------------------------
# Start/Stop (mocked ZMQ)
# ---------------------------------------------------------------------------


class TestStartStop:
    def test_start_requires_zmq(self):
        discovery = ZMQHeartbeatDiscovery()

        with patch.dict('sys.modules', {'zmq': None}):
            discovery.start()
            discovery._listener_thread.join(timeout=1)
            assert discovery.running is False

    def test_start_requires_msgpack(self):
        mock_zmq = MagicMock()
        mock_zmq.Context.return_value = MagicMock()
        mock_zmq.ROUTER = 1
        mock_zmq.POLLIN = 1

        discovery = ZMQHeartbeatDiscovery()

        with (
            patch.dict('sys.modules', {'zmq': mock_zmq}),
            patch.dict('sys.modules', {'msgpack': None}),
        ):
            discovery.start()
            discovery._listener_thread.join(timeout=1)
            assert discovery.running is False

    def test_stop_without_start(self):
        discovery = ZMQHeartbeatDiscovery()
        # Should not raise
        discovery.stop()
        assert discovery._running is False

    def test_start_and_stop_with_real_zmq(self):
        """Test start/stop with real zmq if available."""
        try:
            import msgpack  # noqa: F401
            import zmq  # noqa: F401
        except ImportError:
            pytest.skip('zmq or msgpack not installed')

        # Use a different port to avoid conflicts
        discovery = ZMQHeartbeatDiscovery(port=39999)
        discovery.start()
        assert discovery._running is True
        assert discovery._listener_thread is not None

        discovery.stop()
        assert discovery._running is False


# ---------------------------------------------------------------------------
# Message Handling (unit test internal methods)
# ---------------------------------------------------------------------------


class TestMessageHandling:
    def test_register_prefill_via_internal_method(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        assert discovery.get_prefill_count() == 1
        assert discovery.get_prefill_instances()[0].http_address == '10.0.0.1:8000'

    def test_register_decode_via_internal_method(self):
        discovery = ZMQHeartbeatDiscovery()
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        assert discovery.get_decode_count() == 1
        assert discovery.get_decode_instances()[0].http_address == '10.0.0.2:8000'

    def test_remove_expired_removes_old_instances(self):
        discovery = ZMQHeartbeatDiscovery()
        past = time.time() - 10
        future = time.time() + 100

        discovery._registry.upsert(
            NodeInfo(
                http_address='expired:8000',
                zmq_address='expired:30001',
                role=EngineRole.PREFILL,
                expiration=past,
            )
        )
        discovery._registry.upsert(
            NodeInfo(
                http_address='valid:8000',
                zmq_address='valid:30001',
                role=EngineRole.PREFILL,
                expiration=future,
            )
        )

        discovery._remove_expired()

        assert [node.http_address for node in discovery.get_prefill_instances()] == ['valid:8000']


# ---------------------------------------------------------------------------
# NodeManager Sync (new feature)
# ---------------------------------------------------------------------------


class TestNodeManagerSync:
    """Tests for ZMQ heartbeat discovery to NodeManager sync feature."""

    def test_sync_prefill_to_node_manager(self):
        """Test that prefill instance is synced to node_manager on registration."""
        mock_node_manager = MagicMock()
        models = ['kimi-k2.5']

        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=models,
        )
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        # Verify node_manager.add was called
        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.1:8000'  # URL with http:// prefix
        assert isinstance(call_args[0][1], NodeStatus)
        assert call_args[0][1].role == EngineRole.PREFILL
        assert call_args[0][1].models == models

    def test_sync_decode_to_node_manager(self):
        """Test that decode instance is synced to node_manager on registration."""
        mock_node_manager = MagicMock()
        models = ['kimi-k2.5', 'qwen-72b']

        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=models,
        )
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        # Verify node_manager.add was called
        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.2:8000'
        assert isinstance(call_args[0][1], NodeStatus)
        assert call_args[0][1].role == EngineRole.DECODE
        assert call_args[0][1].models == models

    def test_sync_skipped_when_no_node_manager(self):
        """Test that sync is skipped when node_manager is None."""
        discovery = ZMQHeartbeatDiscovery()  # No node_manager

        # Should not raise
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

    def test_sync_handles_url_with_http_prefix(self):
        """Test that URLs with http:// prefix are handled correctly."""
        mock_node_manager = MagicMock()

        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=['test-model'],
        )
        # URL already has http:// prefix
        discovery._register_prefill('http://10.0.0.1:8000', '10.0.0.1:30001')

        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.1:8000'  # Should not double-prefix

    def test_remove_expired_also_removes_from_node_manager(self):
        """Test that expired instances are removed from node_manager."""
        mock_node_manager = MagicMock()
        discovery = ZMQHeartbeatDiscovery(node_manager=mock_node_manager)

        past = time.time() - 10

        discovery._registry.upsert(
            NodeInfo(
                http_address='expired:8000',
                zmq_address='expired:30001',
                role=EngineRole.PREFILL,
                expiration=past,
            )
        )
        discovery._remove_expired()

        # Verify node_manager.remove was called for expired instance
        mock_node_manager.remove.assert_called_once_with('http://expired:8000')

    def test_sync_error_handled_gracefully(self):
        """Test that sync errors are handled gracefully and logged."""
        mock_node_manager = MagicMock()
        mock_node_manager.add.side_effect = Exception('Node manager error')

        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=['test-model'],
        )
        # Should not raise despite node_manager error
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

    def test_new_registration_fetches_models_before_sync(self):
        """New heartbeat nodes should fetch models before registry upsert and sync."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.return_value = ['Qwen3-4B']
        discovery = ZMQHeartbeatDiscovery(node_manager=mock_node_manager)

        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        mock_node_manager.backend.fetch_models.assert_called_once_with('http://10.0.0.2:8000')
        instances = discovery.get_decode_instances()
        assert len(instances) == 1
        assert instances[0].models == ['Qwen3-4B']

        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.2:8000'
        assert call_args[0][1].models == ['Qwen3-4B']

    def test_new_registration_skips_sync_when_models_unavailable(self):
        """Heartbeat nodes should be skipped when model fetch returns empty."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.return_value = []
        discovery = ZMQHeartbeatDiscovery(node_manager=mock_node_manager)

        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        mock_node_manager.backend.fetch_models.assert_called_once_with('http://10.0.0.1:8000')
        assert discovery.get_prefill_instances() == []
        mock_node_manager.add.assert_not_called()

    def test_failed_registration_is_throttled_until_next_retry_window(self):
        """Repeated heartbeats should not refetch models on every failed registration."""
        mock_node_manager = MagicMock()
        mock_node_manager.backend.fetch_models.return_value = []
        discovery = ZMQHeartbeatDiscovery(node_manager=mock_node_manager)

        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        mock_node_manager.backend.fetch_models.assert_called_once_with('http://10.0.0.2:8000')
        assert discovery.get_decode_instances() == []
