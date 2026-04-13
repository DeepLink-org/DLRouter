"""Tests for service discovery factory helpers."""

from dlrouter.constants import EngineRole, ServiceDiscoveryMode
from dlrouter.core.service_discovery import (
    NodeInfo,
    StaticServiceDiscovery,
    ZMQHeartbeatDiscovery,
    create_service_discovery,
)


class TestCreateServiceDiscoveryFactory:
    def test_static_uses_url_keys_and_normalizes_https(self):
        discovery = create_service_discovery(
            ServiceDiscoveryMode.STATIC,
            {
                'models': ['kimi-k2.5'],
                'prefill_urls': ['https://10.0.0.1:8000'],
                'decode_urls': ['http://10.0.0.2:8000'],
            },
        )

        assert isinstance(discovery, StaticServiceDiscovery)
        assert [node.http_address for node in discovery.get_prefill_instances()] == ['10.0.0.1:8000']
        assert [node.http_address for node in discovery.get_decode_instances()] == ['10.0.0.2:8000']

    def test_static_does_not_accept_legacy_instance_keys(self):
        discovery = create_service_discovery(
            ServiceDiscoveryMode.STATIC,
            {
                'prefill_instances': ['http://10.0.0.1:8000'],
                'decode_instances': ['http://10.0.0.2:8000'],
            },
        )

        assert isinstance(discovery, StaticServiceDiscovery)
        assert discovery.get_prefill_instances() == []
        assert discovery.get_decode_instances() == []

    def test_heartbeat_uses_zmq_config_keys(self):
        discovery = create_service_discovery(
            ServiceDiscoveryMode.HEARTBEAT,
            {
                'zmq_host': '127.0.0.1',
                'zmq_port': 30002,
                'zmq_ping_timeout': 10,
                'models': ['model-a'],
            },
        )

        assert isinstance(discovery, ZMQHeartbeatDiscovery)
        assert discovery._host == '127.0.0.1'
        assert discovery._port == 30002
        assert discovery._ping_timeout == 10
        assert discovery._models == ['model-a']

    def test_static_discovery_exposes_unified_registry_views(self):
        discovery = StaticServiceDiscovery(
            prefill_instances=[
                NodeInfo(
                    http_address='10.0.0.1:8000',
                    role=EngineRole.PREFILL,
                    models=['qwen3-32b'],
                    metadata={'kv_connector': 'mooncake', 'protocol_version': 'v1'},
                )
            ],
            decode_instances=[
                NodeInfo(
                    http_address='10.0.0.2:8000',
                    role=EngineRole.DECODE,
                    models=['qwen3-32b'],
                    metadata={'kv_connector': 'mooncake', 'protocol_version': 'v1'},
                )
            ],
        )

        assert len(discovery.get_all_instances()) == 2
        assert [node.http_address for node in discovery.filter_instances(role=EngineRole.PREFILL)] == ['10.0.0.1:8000']
