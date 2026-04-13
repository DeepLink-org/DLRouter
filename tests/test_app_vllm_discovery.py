"""App-level tests for vLLM discovery inference."""

from dlrouter.api.app import create_app
from dlrouter.config import RouterConfig
from dlrouter.constants import BackendType, ServingStrategy
from dlrouter.core.service_discovery import StaticServiceDiscovery, ZMQHeartbeatDiscovery


def test_create_app_infers_static_vllm_discovery_from_pd_urls():
    app = create_app(
        RouterConfig(
            backend_type=BackendType.VLLM,
            serving_strategy=ServingStrategy.DISTSERVE,
            cache_status=False,
            backend_config={
                'prefill_urls': 'http://10.0.0.1:8200',
                'decode_urls': 'http://10.0.0.2:8200',
            },
        )
    )

    assert isinstance(app.state.service_discovery, StaticServiceDiscovery)


def test_create_app_infers_heartbeat_vllm_discovery_without_pd_urls():
    app = create_app(
        RouterConfig(
            backend_type=BackendType.VLLM,
            serving_strategy=ServingStrategy.DISTSERVE,
            cache_status=False,
            backend_config={
                'zmq_host': '127.0.0.1',
                'zmq_port': 30002,
            },
        )
    )

    assert isinstance(app.state.service_discovery, ZMQHeartbeatDiscovery)
