"""Tests for vLLM pair selection — NodeManager-based."""

import threading
from unittest.mock import MagicMock

from dlrouter.backends.vllm.pair_selection import VLLMPairSelector
from dlrouter.constants import EngineRole
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
    return nm


# -- Basic selection tests --


def test_select_pair_returns_urls_when_candidates_exist() -> None:
    selector = VLLMPairSelector()
    nm = _make_node_manager(
        prefill_urls=['http://10.0.0.1:13700'],
        decode_urls=['http://10.0.0.2:13701'],
    )

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is not None
    prefill_url, decode_url = pair
    assert prefill_url == 'http://10.0.0.1:13700'
    assert decode_url == 'http://10.0.0.2:13701'


def test_select_pair_returns_none_when_no_prefill() -> None:
    selector = VLLMPairSelector()
    nm = _make_node_manager(prefill_urls=[], decode_urls=['http://10.0.0.2:13701'])

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is None


def test_select_pair_returns_none_when_no_decode() -> None:
    selector = VLLMPairSelector()
    nm = _make_node_manager(prefill_urls=['http://10.0.0.1:13700'], decode_urls=[])

    pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')

    assert pair is None


def test_select_pair_returns_none_when_model_not_served() -> None:
    selector = VLLMPairSelector()
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
    selector = VLLMPairSelector()

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
    assert pair[0] == 'http://10.0.0.2:13700'
    assert pair[1] == 'http://10.0.0.1:13701'
    assert mock_router.select_node.call_count == 2


def test_select_pair_rotates_with_round_robin() -> None:
    """Multiple calls with round-robin should work consistently."""
    selector = VLLMPairSelector()
    nm = _make_node_manager(
        prefill_urls=['http://10.0.0.1:13700', 'http://10.0.0.2:13700'],
        decode_urls=['http://10.0.0.1:13701', 'http://10.0.0.2:13701'],
    )

    # Should always return valid pairs
    for _ in range(4):
        pair = selector.select_pair(node_manager=nm, model_name='qwen3-32b')
        assert pair is not None
        assert pair[0].startswith('http://')
        assert pair[1].startswith('http://')
