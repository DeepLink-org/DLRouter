"""App-level tests for LMDeploy DistServe external node registration."""

from dlrouter.api.app import create_app
from dlrouter.config import RouterConfig
from dlrouter.constants import BackendType, EngineRole, ServingStrategy
from dlrouter.models.node import NodeStatus


def test_create_app_allows_lmdeploy_distserve_without_service_discovery() -> None:
    app = create_app(
        RouterConfig(
            backend_type=BackendType.LMDEPLOY,
            serving_strategy=ServingStrategy.DISTSERVE,
            cache_status=False,
        )
    )

    assert app.state.service_discovery is None
    assert app.state.node_manager.serving_strategy is ServingStrategy.DISTSERVE


def test_lmdeploy_distserve_uses_externally_registered_pd_nodes() -> None:
    app = create_app(
        RouterConfig(
            backend_type=BackendType.LMDEPLOY,
            serving_strategy=ServingStrategy.DISTSERVE,
            cache_status=False,
        )
    )
    node_manager = app.state.node_manager

    node_manager.add(
        'http://10.0.0.1:23333',
        NodeStatus(role=EngineRole.PREFILL, models=['qwen']),
    )
    node_manager.add(
        'http://10.0.0.2:23333',
        NodeStatus(role=EngineRole.DECODE, models=['qwen']),
    )

    assert node_manager.get_node_url('qwen', EngineRole.PREFILL) == 'http://10.0.0.1:23333'
    assert node_manager.get_node_url('qwen', EngineRole.DECODE) == 'http://10.0.0.2:23333'
