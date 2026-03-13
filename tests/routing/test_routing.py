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

        All nodes should get default latency 1.0.
        Expected wait = unfinished * 1.0, so select node with min unfinished.
        """
        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=2,  # wait = 2 * 1.0 = 2.0
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # wait = 1 * 1.0 = 1.0 (min)
            ),
        }
        # node2: wait = 1 * 1.0 = 1.0, should always be selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node2:8000'

    def test_mixed_latency_and_empty(self):
        """Test when some nodes have latency data and some don't.

        Empty latency nodes get avg latency of nodes with data.
        Expected wait = unfinished * latency.
        """
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=2,
                latency=deque([0.1, 0.2, 0.1]),  # avg=0.133, wait=2*0.133=0.267
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,
                latency=deque(),  # empty, gets avg=0.133, wait=1*0.133=0.133
            ),
        }
        # node2: wait = 1 * 0.133 = 0.133 (min)
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node2:8000'

    def test_tradeoff_latency_vs_unfinished(self):
        """Test tradeoff between latency and unfinished count.

        A node with lower latency but higher unfinished may have higher
        expected wait than a node with higher latency but lower unfinished.
        """
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=10,
                latency=deque([0.1, 0.1, 0.1]),  # avg=0.1, wait=10*0.1=1.0
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=2,
                latency=deque([0.4, 0.4, 0.4]),  # avg=0.4, wait=2*0.4=0.8 (min)
            ),
        }
        # node1 has lower latency but higher wait time
        # node2 has higher latency but lower wait time -> should be selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node2:8000'

    def test_zero_unfinished_wins(self):
        """Test that zero unfinished always wins (wait=0)."""
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=0,  # wait = 0 * 0.5 = 0 (always min)
                latency=deque([0.5, 0.5, 0.5]),
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,
                latency=deque([0.1, 0.1, 0.1]),  # wait = 1 * 0.1 = 0.1
            ),
        }
        # node1 has zero unfinished -> wait = 0, always selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node1:8000'

    def test_same_wait_random_tie_break(self):
        """Test random tie-break when expected wait times are equal."""
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=2,
                latency=deque([0.5, 0.5, 0.5]),  # wait = 2 * 0.5 = 1.0
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=5,
                latency=deque([0.2, 0.2, 0.2]),  # wait = 5 * 0.2 = 1.0
            ),
        }
        # Both have same expected wait (1.0), should randomly select
        results = set()
        for _ in range(20):
            url = mol.select_node('model-a', cands)
            results.add(url)
        # Should hit both nodes
        assert len(results) == 2

    def test_cold_node_with_high_unfinished(self):
        """Test that cold node with high unfinished is not preferred.

        When a cold node (no latency data) has high unfinished count,
        it should not be selected over warm nodes with lower expected wait.
        """
        from collections import deque

        mol = create_routing_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=1,
                latency=deque([1.0, 1.0, 1.0]),  # avg=1.0, wait=1*1.0=1.0
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=5,
                latency=deque(),  # empty, gets avg=1.0, wait=5*1.0=5.0
            ),
        }
        # node1: wait = 1.0, node2: wait = 5.0
        # node1 should always be selected
        for _ in range(10):
            url = mol.select_node('model-a', cands)
            assert url == 'http://node1:8000'
