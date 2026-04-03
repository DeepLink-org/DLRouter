"""Tests for ZMQServiceDiscovery."""

import time
from unittest.mock import MagicMock, patch

import pytest

from dlrouter.constants import EngineRole
from dlrouter.core.zmq_discovery import ZMQServiceDiscovery
from dlrouter.models.node import NodeStatus


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_values(self):
        discovery = ZMQServiceDiscovery()
        assert discovery._host == '0.0.0.0'
        assert discovery._port == 30001
        assert discovery._ping_timeout == 5
        assert discovery._running is False
        assert discovery._node_manager is None
        assert discovery._models == []

    def test_custom_values(self):
        discovery = ZMQServiceDiscovery(
            host='127.0.0.1',
            port=40001,
            ping_timeout_seconds=10,
        )
        assert discovery._host == '127.0.0.1'
        assert discovery._port == 40001
        assert discovery._ping_timeout == 10

    def test_port_zero_raises(self):
        with pytest.raises(ValueError, match='Port cannot be 0'):
            ZMQServiceDiscovery(port=0)

    def test_init_with_node_manager_and_models(self):
        """Test initialization with node_manager and models (new feature)."""
        mock_node_manager = MagicMock()
        models = ['kimi-k2.5', 'qwen-72b']

        discovery = ZMQServiceDiscovery(
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
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        # Manually register a prefill instance
        with discovery._prefill_lock:
            discovery._prefill_instances['10.0.0.1:8000'] = (
                '10.0.0.1:30001',
                future_time,
            )

        assert discovery.get_prefill_count() == 1
        instances = discovery.prefill_instances
        assert '10.0.0.1:8000' in instances
        assert instances['10.0.0.1:8000'][0] == '10.0.0.1:30001'

    def test_register_decode_instance(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        # Manually register a decode instance
        with discovery._decode_lock:
            discovery._decode_instances['10.0.0.2:8000'] = (
                '10.0.0.2:30001',
                future_time,
            )

        assert discovery.get_decode_count() == 1
        instances = discovery.decode_instances
        assert '10.0.0.2:8000' in instances
        assert instances['10.0.0.2:8000'][0] == '10.0.0.2:30001'

    def test_expired_instances_removed(self):
        discovery = ZMQServiceDiscovery()
        past_time = time.time() - 10  # Already expired

        with discovery._prefill_lock:
            discovery._prefill_instances['expired:8000'] = (
                'expired:30001',
                past_time,
            )

        # get_prefill_count triggers cleanup
        assert discovery.get_prefill_count() == 0


# ---------------------------------------------------------------------------
# Instance Selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_select_prefill_instance_round_robin(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        # Add two prefill instances
        with discovery._prefill_lock:
            discovery._prefill_instances['10.0.0.1:8000'] = (
                '10.0.0.1:30001',
                future_time,
            )
            discovery._prefill_instances['10.0.0.2:8000'] = (
                '10.0.0.2:30001',
                future_time,
            )

        result = discovery.select_prefill_instance()
        assert result is not None
        assert result[0] in ['10.0.0.1:8000', '10.0.0.2:8000']

    def test_select_decode_instance_round_robin(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        # Add two decode instances
        with discovery._decode_lock:
            discovery._decode_instances['10.0.0.3:8000'] = (
                '10.0.0.3:30001',
                future_time,
            )
            discovery._decode_instances['10.0.0.4:8000'] = (
                '10.0.0.4:30001',
                future_time,
            )

        result = discovery.select_decode_instance()
        assert result is not None
        assert result[0] in ['10.0.0.3:8000', '10.0.0.4:8000']

    def test_select_prefill_returns_none_when_empty(self):
        discovery = ZMQServiceDiscovery()
        assert discovery.select_prefill_instance() is None

    def test_select_decode_returns_none_when_empty(self):
        discovery = ZMQServiceDiscovery()
        assert discovery.select_decode_instance() is None

    def test_select_pd_pair(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        # Add instances
        with discovery._prefill_lock:
            discovery._prefill_instances['10.0.0.1:8000'] = (
                '10.0.0.1:30001',
                future_time,
            )
        with discovery._decode_lock:
            discovery._decode_instances['10.0.0.2:8000'] = (
                '10.0.0.2:30001',
                future_time,
            )

        result = discovery.select_pd_pair()
        assert result is not None
        prefill, decode = result
        assert prefill == ('10.0.0.1:8000', '10.0.0.1:30001')
        assert decode == ('10.0.0.2:8000', '10.0.0.2:30001')

    def test_select_pd_pair_returns_none_without_prefill(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        with discovery._decode_lock:
            discovery._decode_instances['10.0.0.2:8000'] = (
                '10.0.0.2:30001',
                future_time,
            )

        assert discovery.select_pd_pair() is None

    def test_select_pd_pair_returns_none_without_decode(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        with discovery._prefill_lock:
            discovery._prefill_instances['10.0.0.1:8000'] = (
                '10.0.0.1:30001',
                future_time,
            )

        assert discovery.select_pd_pair() is None


# ---------------------------------------------------------------------------
# Request ID Building
# ---------------------------------------------------------------------------


class TestRequestIdBuilding:
    def test_build_request_id_format(self):
        discovery = ZMQServiceDiscovery()
        request_id = discovery.build_request_id(
            prefill_zmq_addr='10.0.0.1:30001',
            decode_zmq_addr='10.0.0.2:30001',
            base_id='abc123',
        )

        assert '___prefill_addr_10.0.0.1:30001' in request_id
        assert '___decode_addr_10.0.0.2:30001' in request_id
        assert 'abc123' in request_id

    def test_build_request_id_uniqueness(self):
        discovery = ZMQServiceDiscovery()
        id1 = discovery.build_request_id('p1', 'd1', 'uuid1')
        id2 = discovery.build_request_id('p1', 'd1', 'uuid2')
        assert id1 != id2


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_get_status(self):
        discovery = ZMQServiceDiscovery()
        future_time = time.time() + 100

        with discovery._prefill_lock:
            discovery._prefill_instances['10.0.0.1:8000'] = (
                '10.0.0.1:30001',
                future_time,
            )
        with discovery._decode_lock:
            discovery._decode_instances['10.0.0.2:8000'] = (
                '10.0.0.2:30001',
                future_time,
            )

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
        discovery = ZMQServiceDiscovery()

        with patch.dict('sys.modules', {'zmq': None}), pytest.raises(ImportError):
            discovery.start()

    def test_start_requires_msgpack(self):
        mock_zmq = MagicMock()
        mock_zmq.Context.return_value = MagicMock()
        mock_zmq.ROUTER = 1
        mock_zmq.POLLIN = 1

        discovery = ZMQServiceDiscovery()

        with (
            patch.dict('sys.modules', {'zmq': mock_zmq}),
            patch.dict('sys.modules', {'msgpack': None}),
            pytest.raises(ImportError),
        ):
            discovery.start()

    def test_stop_without_start(self):
        discovery = ZMQServiceDiscovery()
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
        discovery = ZMQServiceDiscovery(port=39999)
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
        discovery = ZMQServiceDiscovery()
        expiration = time.time() + 100

        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001', expiration)

        assert discovery.get_prefill_count() == 1
        assert '10.0.0.1:8000' in discovery.prefill_instances

    def test_register_decode_via_internal_method(self):
        discovery = ZMQServiceDiscovery()
        expiration = time.time() + 100

        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001', expiration)

        assert discovery.get_decode_count() == 1
        assert '10.0.0.2:8000' in discovery.decode_instances

    def test_remove_expired_removes_old_instances(self):
        discovery = ZMQServiceDiscovery()
        past = time.time() - 10
        future = time.time() + 100

        with discovery._prefill_lock:
            discovery._prefill_instances['expired:8000'] = ('expired:30001', past)
            discovery._prefill_instances['valid:8000'] = ('valid:30001', future)
            discovery._remove_expired(discovery._prefill_instances)

        assert 'expired:8000' not in discovery._prefill_instances
        assert 'valid:8000' in discovery._prefill_instances


# ---------------------------------------------------------------------------
# NodeManager Sync (new feature)
# ---------------------------------------------------------------------------


class TestNodeManagerSync:
    """Tests for ZMQ discovery to NodeManager sync feature."""

    def test_sync_prefill_to_node_manager(self):
        """Test that prefill instance is synced to node_manager on registration."""
        mock_node_manager = MagicMock()
        models = ['kimi-k2.5']

        discovery = ZMQServiceDiscovery(
            node_manager=mock_node_manager,
            models=models,
        )
        expiration = time.time() + 100

        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001', expiration)

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

        discovery = ZMQServiceDiscovery(
            node_manager=mock_node_manager,
            models=models,
        )
        expiration = time.time() + 100

        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001', expiration)

        # Verify node_manager.add was called
        mock_node_manager.add.assert_called_once()
        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.2:8000'
        assert isinstance(call_args[0][1], NodeStatus)
        assert call_args[0][1].role == EngineRole.DECODE
        assert call_args[0][1].models == models

    def test_sync_skipped_when_no_node_manager(self):
        """Test that sync is skipped when node_manager is None."""
        discovery = ZMQServiceDiscovery()  # No node_manager
        expiration = time.time() + 100

        # Should not raise
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001', expiration)
        discovery._register_decode('10.0.0.2:8000', '10.0.0.2:30001', expiration)

    def test_sync_handles_url_with_http_prefix(self):
        """Test that URLs with http:// prefix are handled correctly."""
        mock_node_manager = MagicMock()

        discovery = ZMQServiceDiscovery(
            node_manager=mock_node_manager,
            models=['test-model'],
        )
        expiration = time.time() + 100

        # URL already has http:// prefix
        discovery._register_prefill('http://10.0.0.1:8000', '10.0.0.1:30001', expiration)

        call_args = mock_node_manager.add.call_args
        assert call_args[0][0] == 'http://10.0.0.1:8000'  # Should not double-prefix

    def test_remove_expired_also_removes_from_node_manager(self):
        """Test that expired instances are removed from node_manager."""
        mock_node_manager = MagicMock()
        discovery = ZMQServiceDiscovery(node_manager=mock_node_manager)

        past = time.time() - 10

        with discovery._prefill_lock:
            discovery._prefill_instances['expired:8000'] = ('expired:30001', past)
            discovery._remove_expired(discovery._prefill_instances)

        # Verify node_manager.remove was called for expired instance
        mock_node_manager.remove.assert_called_once_with('http://expired:8000')

    def test_sync_error_handled_gracefully(self):
        """Test that sync errors are handled gracefully and logged."""
        mock_node_manager = MagicMock()
        mock_node_manager.add.side_effect = Exception('Node manager error')

        discovery = ZMQServiceDiscovery(
            node_manager=mock_node_manager,
            models=['test-model'],
        )
        expiration = time.time() + 100

        # Should not raise despite node_manager error
        discovery._register_prefill('10.0.0.1:8000', '10.0.0.1:30001', expiration)
