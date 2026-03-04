"""Tests for routing strategies."""

from dlrouter.constants import RoutingStrategy
from dlrouter.models.node import NodeStatus
from dlrouter.routing.factory import create_routing_strategy


def _make_candidates():
    """Create test candidate nodes."""
    return {
        'http://node1:8000': NodeStatus(
            models=['model-a', 'model-b'],
            speed=10.0,
            unfinished=2,
        ),
        'http://node2:8000': NodeStatus(
            models=['model-a'],
            speed=20.0,
            unfinished=1,
        ),
        'http://node3:8000': NodeStatus(
            models=['model-b'],
            speed=5.0,
            unfinished=0,
        ),
    }


class TestRoundRobin:
    def test_basic(self):
        rr = create_routing_strategy(RoutingStrategy.ROUND_ROBIN)
        cands = _make_candidates()
        results = set()
        for _ in range(10):
            url = rr.select_node('model-a', cands)
            assert url is not None
            results.add(url)
        # Should hit both nodes serving model-a
        assert len(results) == 2

    def test_no_match(self):
        rr = create_routing_strategy(RoutingStrategy.ROUND_ROBIN)
        cands = _make_candidates()
        url = rr.select_node('nonexistent', cands)
        assert url is None


class TestRandom:
    def test_basic(self):
        rand = create_routing_strategy(RoutingStrategy.RANDOM)
        cands = _make_candidates()
        for _ in range(10):
            url = rand.select_node('model-a', cands)
            assert url in [
                'http://node1:8000',
                'http://node2:8000',
            ]


class TestConsistentHash:
    def test_same_key_same_node(self):
        ch = create_routing_strategy(RoutingStrategy.CONSISTENT_HASH)
        cands = _make_candidates()
        for _ in range(10):
            url1 = ch.select_node('model-a', cands, 'user-123')
            url2 = ch.select_node('model-a', cands, 'user-123')
            assert url1 == url2
            assert url1 is not None

    def test_different_keys(self):
        ch = create_routing_strategy(RoutingStrategy.CONSISTENT_HASH)
        cands = _make_candidates()
        # Different keys may or may not hit
        # different nodes but should return valid urls
        url1 = ch.select_node('model-a', cands, 'user-1')
        url2 = ch.select_node('model-a', cands, 'user-2')
        assert url1 is not None
        assert url2 is not None


class TestMinExpectedLatency:
    def test_picks_lowest_latency(self):
        mel = create_routing_strategy(RoutingStrategy.MIN_EXPECTED_LATENCY)
        cands = _make_candidates()
        # node2: 1/20 = 0.05
        # node1: 2/10 = 0.2
        url = mel.select_node('model-a', cands)
        assert url == 'http://node2:8000'


class TestMinObservedLatency:
    def test_basic(self):
        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = _make_candidates()
        # Both have empty latency -> inf
        # Should still return something
        url = mol.select_node('model-a', cands)
        assert url is not None
