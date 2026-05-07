"""NodeManager-based tests for shared PD pair selection."""

import threading
from unittest.mock import MagicMock

from dlrouter.backends.pd import PDPair, PDPairSelector
from dlrouter.constants import EngineRole, RoutingStrategy, ServingStrategy
from dlrouter.core.node_manager import NodeManager
from dlrouter.models.node import NodeStatus
from dlrouter.routing.round_robin import RoundRobinStrategy


def _make_node_manager(
    prefill_urls: list[str],
    decode_urls: list[str],
    *,
    model: str = 'qwen3-32b',
    unfinished: dict[str, int] | None = None,
) -> MagicMock:
    """Build a mock NodeManager with real nodes dict and a real router."""
    unfinished = unfinished or {}
    nodes: dict[str, NodeStatus] = {}
    for url in prefill_urls:
        nodes[url] = NodeStatus(
            role=EngineRole.PREFILL,
            models=[model],
            unfinished=unfinished.get(url, 0),
        )
    for url in decode_urls:
        nodes[url] = NodeStatus(
            role=EngineRole.DECODE,
            models=[model],
            unfinished=unfinished.get(url, 0),
        )
    nm = MagicMock()
    nm.nodes = nodes
    nm._lock = threading.RLock()
    nm._router = RoundRobinStrategy()

    # Wire up prefill_nodes / decode_nodes properties
    nm.prefill_nodes = {url: st for url, st in nodes.items() if st.role == EngineRole.PREFILL}
    nm.decode_nodes = {url: st for url, st in nodes.items() if st.role == EngineRole.DECODE}

    def get_node_url(model_name: str, role: EngineRole = EngineRole.HYBRID, request_key: str | None = None):
        candidates = {url: st for url, st in nodes.items() if st.role == role}
        return nm._router.select_node(model_name, candidates, request_key)

    nm.get_node_url.side_effect = get_node_url
    return nm


def test_node_manager_add_returns_false_when_backend_registration_fails() -> None:
    backend = MagicMock()
    backend.register_node.side_effect = RuntimeError('register failed')
    manager = NodeManager(
        backend=backend,
        routing_strategy=RoutingStrategy.ROUND_ROBIN,
        serving_strategy=ServingStrategy.HYBRID,
        cache_status=False,
    )
    manager._save_config = MagicMock()

    added = manager.add('http://node:8000')

    assert added is False
    assert 'http://node:8000' not in manager.nodes
    manager._save_config.assert_not_called()


# -- Basic selection tests --


def test_select_pair_returns_urls_when_candidates_exist() -> None:
    selector = PDPairSelector()
    nm = _make_node_manager(
        prefill_urls=['http://10.0.0.1:13700'],
        decode_urls=['http://10.0.0.2:13701'],
    )

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is not None
    assert pair.prefill_url == 'http://10.0.0.1:13700'
    assert pair.decode_url == 'http://10.0.0.2:13701'


def test_select_pair_returns_none_when_no_prefill() -> None:
    selector = PDPairSelector()
    nm = _make_node_manager(prefill_urls=[], decode_urls=['http://10.0.0.2:13701'])

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is None


def test_select_pair_returns_none_when_no_decode() -> None:
    selector = PDPairSelector()
    nm = _make_node_manager(prefill_urls=['http://10.0.0.1:13700'], decode_urls=[])

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is None


def test_select_pair_returns_none_when_model_not_served() -> None:
    selector = PDPairSelector()
    nm = _make_node_manager(
        prefill_urls=['http://10.0.0.1:13700'],
        decode_urls=['http://10.0.0.2:13701'],
        model='qwen3-32b',
    )

    pair = selector.select_pair(node_manager=nm, model_name='llama3')

    assert pair is None


# -- Routing strategy tests --


def test_select_pair_uses_node_manager_router() -> None:
    """Verify selection delegates to NodeManager's routing strategy."""
    selector = PDPairSelector()

    # Mock router that always picks the second prefill node
    mock_router = MagicMock()
    mock_router.select_node.side_effect = lambda model, candidates, key=None: (
        'http://10.0.0.2:13700' if any('10.0.0.2:13700' in u for u in candidates) else next(iter(candidates))
    )

    nm = _make_node_manager(
        prefill_urls=['http://10.0.0.1:13700', 'http://10.0.0.2:13700'],
        decode_urls=['http://10.0.0.1:13701'],
    )
    nm._router = mock_router

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is not None
    assert pair.prefill_url == 'http://10.0.0.2:13700'
    assert pair.decode_url == 'http://10.0.0.1:13701'
    assert mock_router.select_node.call_count == 2


def test_select_pair_passes_request_key_to_node_manager() -> None:
    selector = PDPairSelector()
    nm = MagicMock()
    nm.get_node_url.side_effect = ['http://prefill:8000', 'http://decode:8000']

    pair = selector.select_pair(
        node_manager=nm,
        model_name='qwen3-32b',
        request_key='session-123',
    )

    assert pair == PDPair('http://prefill:8000', 'http://decode:8000')
    assert nm.get_node_url.call_args_list[0].kwargs['request_key'] == 'session-123'
    assert nm.get_node_url.call_args_list[1].kwargs['request_key'] == 'session-123'


def test_select_pair_rotates_with_round_robin() -> None:
    """Multiple calls with round-robin should work consistently."""
    selector = PDPairSelector()
    nm = _make_node_manager(
        prefill_urls=['http://10.0.0.1:13700', 'http://10.0.0.2:13700'],
        decode_urls=['http://10.0.0.1:13701', 'http://10.0.0.2:13701'],
    )

    # Should always return valid pairs
    for _ in range(4):
        pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')
        assert pair is not None
        assert pair.prefill_url.startswith('http://')
        assert pair.decode_url.startswith('http://')
