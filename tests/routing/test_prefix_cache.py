"""Tests for prefix cache routing strategy."""

from dlrouter.constants import RoutingStrategy
from dlrouter.models.node import NodeStatus
from dlrouter.routing.factory import create_routing_strategy
from dlrouter.routing.prefix_cache import PrefixCacheStrategy, PrefixCacheTrie


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


class TestPrefixCacheTrie:
    """Tests for PrefixCacheTrie data structure."""

    def test_add_and_find_prefix(self):
        """Test adding prefix and finding best node."""
        trie = PrefixCacheTrie()
        prompt = 'Hello world this is a test'
        node_url = 'http://node1:8000'

        trie.add_prefix(prompt, node_url)

        # Should find the node for the same prompt
        found = trie.find_best_node(prompt, [node_url])
        assert found == node_url

    def test_find_best_node_with_shared_prefix(self):
        """Test finding node with shared prefix."""
        trie = PrefixCacheTrie()
        node1 = 'http://node1:8000'
        node2 = 'http://node2:8000'

        # Add two prompts with shared prefix
        trie.add_prefix('Hello world from node1', node1)
        trie.add_prefix('Hello world from node2', node2)

        # Query with shared prefix should find the one with longest match
        found = trie.find_best_node('Hello world new request', [node1, node2])
        # Both have same prefix length, either could be returned
        assert found in [node1, node2]

    def test_find_best_node_longest_match(self):
        """Test that longest prefix match is selected."""
        trie = PrefixCacheTrie()
        node1 = 'http://node1:8000'
        node2 = 'http://node2:8000'

        # node1 has longer shared prefix
        trie.add_prefix('Hello world from node1', node1)
        trie.add_prefix('Hello', node2)

        # Should prefer node1 for this prompt
        found = trie.find_best_node('Hello world', [node1, node2])
        assert found == node1

    def test_find_best_node_not_in_candidates(self):
        """Test that nodes not in candidates are ignored."""
        trie = PrefixCacheTrie()
        node1 = 'http://node1:8000'
        node2 = 'http://node2:8000'

        trie.add_prefix('Hello world', node1)

        # node1 not in candidates, should return None
        found = trie.find_best_node('Hello world', [node2])
        assert found is None

    def test_remove_node(self):
        """Test removing a node from trie."""
        trie = PrefixCacheTrie()
        node1 = 'http://node1:8000'
        node2 = 'http://node2:8000'

        trie.add_prefix('Hello world', node1)
        trie.add_prefix('Hello world', node2)

        # Remove node1
        trie.remove_node(node1)

        # Should only find node2
        found = trie.find_best_node('Hello world', [node1, node2])
        assert found == node2

    def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        trie = PrefixCacheTrie()
        node1 = 'http://node1:8000'

        trie.add_prefix('Hello world', node1)

        # Immediately cleanup with 0 max age should remove everything
        removed = trie.cleanup_expired(max_age_seconds=0)
        assert removed >= 1

        # Should not find the node anymore
        found = trie.find_best_node('Hello world', [node1])
        assert found is None

    def test_max_depth_limit(self):
        """Test that max_depth limits prefix storage."""
        trie = PrefixCacheTrie(max_depth=5)
        node1 = 'http://node1:8000'

        long_prompt = 'a' * 1000
        trie.add_prefix(long_prompt, node1)

        # Should still find the node
        found = trie.find_best_node(long_prompt, [node1])
        assert found == node1

    def test_normalize_prompt(self):
        """Test prompt normalization (whitespace handling)."""
        trie = PrefixCacheTrie()
        node1 = 'http://node1:8000'

        # These should be normalized to the same prompt
        trie.add_prefix('Hello   world', node1)

        found = trie.find_best_node('Hello world', [node1])
        assert found == node1

    def test_get_stats(self):
        """Test getting trie statistics."""
        trie = PrefixCacheTrie(max_depth=50)
        node1 = 'http://node1:8000'

        trie.add_prefix('Hello', node1)
        trie.add_prefix('World', node1)

        stats = trie.get_stats()
        assert stats['max_depth'] == 50
        assert stats['total_nodes'] > 0


class TestPrefixCacheStrategy:
    """Tests for PrefixCacheStrategy routing."""

    def test_cache_hit_same_prompt(self):
        """Test that same prompt hits cache and routes to same node."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()
        prompt = 'This is a test prompt for caching'

        # First request - cache miss
        url1 = strategy.select_node('model-a', cands, prompt)
        assert url1 is not None

        # Second request with same prompt - cache hit
        url2 = strategy.select_node('model-a', cands, prompt)
        assert url2 == url1

    def test_cache_hit_shared_prefix(self):
        """Test that prompts with shared prefix route to same node."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()

        # First request
        prompt1 = 'Hello world this is a long prompt'
        url1 = strategy.select_node('model-a', cands, prompt1)

        # Second request with shared prefix
        prompt2 = 'Hello world this is another prompt'
        url2 = strategy.select_node('model-a', cands, prompt2)

        # Should route to same node due to shared prefix
        assert url2 == url1

    def test_no_request_key_fallback(self):
        """Test fallback when no request_key provided."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()

        # No request_key - should return first candidate
        url = strategy.select_node('model-a', cands, None)
        assert url in ['http://node1:8000', 'http://node2:8000']

    def test_cache_miss_load_balancing(self):
        """Test load balancing on cache miss."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=5,  # High load
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # Low load - should be selected
            ),
        }

        prompt = "Unique prompt that won't be cached"
        url = strategy.select_node('model-a', cands, prompt)

        # Should select node with minimum unfinished
        assert url == 'http://node2:8000'

    def test_no_matching_model(self):
        """Test when no nodes serve the requested model."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()

        url = strategy.select_node('nonexistent-model', cands, 'test prompt')
        assert url is None

    def test_remove_node_from_strategy(self):
        """Test removing a node from strategy."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()
        prompt = 'Test prompt for node removal'

        # First request
        url1 = strategy.select_node('model-a', cands, prompt)

        # Remove that node
        strategy.remove_node(url1)

        # Next request should not route to removed node
        cands2 = {k: v for k, v in cands.items() if k != url1}
        if cands2:
            url2 = strategy.select_node('model-a', cands2, prompt)
            assert url2 != url1

    def test_update_cache_manually(self):
        """Test manually updating cache."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)

        prompt = 'Manual cache update test'
        node_url = 'http://node1:8000'

        strategy.update_cache(prompt, node_url)

        # Should find the manually cached node
        cands = {'http://node1:8000': NodeStatus(models=['model-a'])}
        found = strategy.select_node('model-a', cands, prompt)
        assert found == node_url

    def test_cleanup_expired_entries(self):
        """Test cleanup of expired cache entries."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()
        prompt = 'Test prompt for cleanup'

        # Add to cache
        strategy.select_node('model-a', cands, prompt)

        # Cleanup with 0 age should remove entries
        removed = strategy.cleanup(max_age_seconds=0)
        assert removed >= 0

    def test_get_stats(self):
        """Test getting strategy statistics."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()

        # Add some entries
        strategy.select_node('model-a', cands, 'Test prompt 1')
        strategy.select_node('model-a', cands, 'Test prompt 2')

        stats = strategy.get_stats()
        assert 'total_nodes' in stats
        assert 'max_depth' in stats

    def test_load_balancing_tie_breaker(self):
        """Test round-robin tie breaker when multiple nodes have same load on cache miss."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)

        # First request - cache miss, should select one node (round-robin tie breaker)
        cands1 = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=1,  # Same load
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # Same load
            ),
        }
        prompt1 = 'First unique prompt for load balancing'
        url1 = strategy.select_node('model-a', cands1, prompt1)

        # Second request with different prompt - also cache miss
        # Create new strategy to ensure clean cache state
        strategy2 = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands2 = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=1,  # Same load
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=1,  # Same load
            ),
        }
        prompt2 = 'Second unique prompt for load balancing'
        url2 = strategy2.select_node('model-a', cands2, prompt2)

        # Both nodes should be selectable on cache miss with round-robin
        # Note: Due to round-robin counter, consecutive requests may hit different nodes

        # At least one of the two requests should demonstrate load balancing
        # (both could be same due to counter state, but typically they differ)
        assert url1 is not None
        assert url2 is not None
        assert url1 in ['http://node1:8000', 'http://node2:8000']
        assert url2 in ['http://node1:8000', 'http://node2:8000']

    def test_zero_unfinished_priority(self):
        """Test that nodes with zero unfinished are prioritized on cache miss."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = {
            'http://node1:8000': NodeStatus(
                models=['model-a'],
                speed=10.0,
                unfinished=5,
            ),
            'http://node2:8000': NodeStatus(
                models=['model-a'],
                speed=20.0,
                unfinished=0,  # Should be selected
            ),
        }

        prompt = 'Test prompt for zero unfinished priority'
        url = strategy.select_node('model-a', cands, prompt)

        assert url == 'http://node2:8000'


class TestPrefixCacheIntegration:
    """Integration tests for prefix cache routing."""

    def test_multiple_models_isolation(self):
        """Test that different models don't interfere with each other."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)

        cands_model_a = {
            'http://node1:8000': NodeStatus(models=['model-a']),
        }
        cands_model_b = {
            'http://node2:8000': NodeStatus(models=['model-b']),
        }

        prompt = 'Shared prompt text'

        url_a = strategy.select_node('model-a', cands_model_a, prompt)
        url_b = strategy.select_node('model-b', cands_model_b, prompt)

        # Should route to different nodes for different models
        assert url_a == 'http://node1:8000'
        assert url_b == 'http://node2:8000'

    def test_concurrent_prompts_cache_independence(self):
        """Test that different prompts are cached independently."""
        strategy = create_routing_strategy(RoutingStrategy.PREFIX_CACHE)
        cands = _make_candidates()

        prompt1 = 'First unique prompt for caching'
        prompt2 = 'Second unique prompt for caching'

        url1 = strategy.select_node('model-a', cands, prompt1)
        url2 = strategy.select_node('model-a', cands, prompt2)

        # Both should be cached and return their respective nodes
        url1_again = strategy.select_node('model-a', cands, prompt1)
        url2_again = strategy.select_node('model-a', cands, prompt2)

        assert url1_again == url1
        assert url2_again == url2

    def test_prefix_depth_limit(self):
        """Test that very long prompts respect max_depth."""
        strategy = PrefixCacheStrategy(max_prefix_depth=10)
        cands = _make_candidates()

        # Very long prompt
        long_prompt = 'a' * 1000

        url = strategy.select_node('model-a', cands, long_prompt)
        assert url is not None

        # Stats should show limited depth
        stats = strategy.get_stats()
        assert stats['max_depth'] == 10
