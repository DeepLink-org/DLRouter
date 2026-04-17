"""Tests for ZMQHeartbeatDiscovery."""

from unittest.mock import MagicMock, patch

import pytest

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import ZMQHeartbeatDiscovery
from dlrouter.models.node import NodeStatus


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_values(self):
        discovery = ZMQHeartbeatDiscovery()
        assert discovery._host == '0.0.0.0'
        assert discovery._port == 30001
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

    def test_port_zero_raises(self):
        with pytest.raises(ValueError, match='Port cannot be 0'):
            ZMQHeartbeatDiscovery(port=0)

    def test_init_with_node_manager_and_models(self):
        """Test initialization with node_manager and models."""
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
    def test_register_prefill_syncs_to_node_manager(self):
        mock_node_manager = MagicMock()
        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=['kimi-k2.5'],
        )
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.1:8000'
        assert isinstance(call_args[0][1], NodeStatus)
        assert call_args[0][1].role == EngineRole.PREFILL
        assert call_args[0][1].zmq_address == '10.0.0.1:30001'
        assert call_args[0][1].models == ['kimi-k2.5']

    def test_register_decode_syncs_to_node_manager(self):
        mock_node_manager = MagicMock()
        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=['kimi-k2.5'],
        )
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001')

        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.2:8000'
        assert call_args[0][1].role == EngineRole.DECODE
        assert call_args[0][1].zmq_address == '10.0.0.2:30001'

    def test_duplicate_heartbeat_does_not_re_register(self):
        """Second heartbeat from same address should not call add again."""
        mock_node_manager = MagicMock()
        discovery = ZMQHeartbeatDiscovery(
            node_manager=mock_node_manager,
            models=['kimi-k2.5'],
        )
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001')

        # Only one add call despite two heartbeats
        mock_node_manager.add.assert_called_once()

    def test_handle_zmq_message_preserves_optional_metadata(self):
        discovery = ZMQHeartbeatDiscovery(
            node_manager=MagicMock(),
            models=['test-model'],
        )
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

        # Should have registered the prefill
        discovery._node_manager.add.assert_called_once()


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
# NodeManager Sync
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

        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.1:8000'
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


# ---------------------------------------------------------------------------
# Legacy request-id cleanup
# ---------------------------------------------------------------------------


class TestLegacyRequestIdCleanup:
    def test_build_request_id_is_no_longer_exposed(self):
        discovery = ZMQHeartbeatDiscovery()
        assert not hasattr(discovery, 'build_request_id')
