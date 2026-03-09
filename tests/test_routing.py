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
        # Both have empty latency -> default 1.0
        # Should still return something
        url = mol.select_node('model-a', cands)
        assert url is not None

    def test_cold_start_all_empty_latency(self):
        """Test when all nodes have empty latency (cold start).

        All nodes should get DEFAULT_LATENCY (1.0), so all are candidates.
        Then select by minimum unfinished.
        """
        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=2,
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # min unfinished
            ),
        }
        # All have default latency 1.0, then select by unfinished
        # node2 has unfinished=1, should always be selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node2:8000'

    def test_mixed_latency_and_empty(self):
        """Test when some nodes have latency data and some don't.

        Empty latency nodes get DEFAULT_LATENCY=1.0.
        If real latency < 1.5, nodes with data are preferred.
        """
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=2,
                latency=deque([0.1, 0.2, 0.1]),  # avg = 0.133
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,
                latency=deque(),  # empty, gets default 1.0
            ),
        }
        # node1 has latency 0.133, node2 has default 1.0
        # min_lat = 0.133, threshold = 0.133 * 1.5 = 0.2
        # Only node1 (0.133 < 0.2) is selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node1:8000'

    def test_similar_latency_select_by_unfinished(self):
        """Test selecting by unfinished when latencies are similar.

        Among nodes with similar latency (within 50%),
        should select the one with minimum unfinished count.
        """
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=5,  # higher unfinished
                latency=deque([0.5, 0.5, 0.5]),  # avg = 0.5
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # lowest unfinished
                latency=deque([0.7, 0.7, 0.7]),  # avg = 0.7 (< 0.5 * 1.5 = 0.75)
            ),
            'http://node3:8000': NodeStatus(
                models=['model-a'],
                speed=15.0,
                unfinished=0,
                latency=deque([1.0, 1.0, 1.0]),  # avg = 1.0 (> 0.75, excluded)
            ),
        }
        # min_lat = 0.5, threshold = 0.75
        # node1 (0.5) and node2 (0.7) are within similar latency range
        # node2 has unfinished=1 (min), so should always be selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node2:8000'

    def test_same_unfinished_random_tie_break(self):
        """Test random tie-break when both latency and unfinished are equal."""
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=1,
                latency=deque([0.5, 0.5, 0.5]),  # avg = 0.5
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # same unfinished
                latency=deque([0.6, 0.6, 0.6]),  # avg = 0.6 (< 0.5 * 1.5 = 0.75)
            ),
        }
        # Both have similar latency and same unfinished count
        # Should randomly select from both
        results = set()
        for _ in range(20):
            url = mol.select_node('model-a', cands)
            results.add(url)
        # Should hit both nodes
        assert len(results) == 2
