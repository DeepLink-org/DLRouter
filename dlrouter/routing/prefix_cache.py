"""Prefix cache aware routing strategy.

This module implements a prefix-aware routing strategy using a Trie data structure.
Similar to vLLM-router's implementation, it routes requests with shared prefixes
to the same backend node to maximize KV cache utilization.
"""

import threading
import time
from typing import Optional

from dlrouter.logger import get_logger
from dlrouter.models.node import NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy

logger = get_logger('dlrouter.prefix_cache')


class TrieNode:
    """A node in the prefix Trie."""

    def __init__(self):
        self.children: dict = {}
        self.nodes: dict = {}
        self.is_end = False


class PrefixCacheTrie:
    """Prefix cache using Trie data structure."""

    def __init__(self, max_depth: int = 100):
        self._root = TrieNode()
        self._max_depth = max_depth
        self._lock = threading.RLock()

    def _normalize_prompt(self, prompt: str) -> str:
        return ' '.join(prompt.split())

    def add_prefix(self, prompt: str, node_url: str) -> None:
        normalized = self._normalize_prompt(prompt)
        if not normalized:
            return
        with self._lock:
            current = self._root
            for i, char in enumerate(normalized):
                if i >= self._max_depth:
                    break
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
                current.nodes[node_url] = time.time()
            current.is_end = True

    def find_best_node(self, prompt: str, candidate_nodes: list) -> Optional[str]:
        normalized = self._normalize_prompt(prompt)
        if not normalized or not candidate_nodes:
            return None
        candidate_set = set(candidate_nodes)
        best_node: Optional[str] = None
        best_depth = 0
        with self._lock:
            current = self._root
            for depth, char in enumerate(normalized):
                if depth >= self._max_depth:
                    break
                if char not in current.children:
                    break
                current = current.children[char]
                for node_url in list(current.nodes.keys()):
                    if node_url in candidate_set:
                        if depth + 1 > best_depth:
                            best_depth = depth + 1
                            best_node = node_url
            if best_node:
                current.nodes[best_node] = time.time()
        return best_node

    def remove_node(self, node_url: str) -> None:
        with self._lock:
            self._remove_node_recursive(self._root, node_url)

    def _remove_node_recursive(self, node: TrieNode, node_url: str) -> bool:
        node.nodes.pop(node_url, None)
        keys_to_remove = []
        for key, child in node.children.items():
            if self._remove_node_recursive(child, node_url):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del node.children[key]
        return not node.children and not node.nodes and not node.is_end

    def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        cutoff_time = time.time() - max_age_seconds
        removed_count = [0]
        with self._lock:
            self._cleanup_recursive(self._root, cutoff_time, removed_count)
        return removed_count[0]

    def _cleanup_recursive(self, node: TrieNode, cutoff_time: float, removed_count: list) -> bool:
        expired = [url for url, t in node.nodes.items() if t < cutoff_time]
        for url in expired:
            del node.nodes[url]
            removed_count[0] += 1
        keys_to_remove = []
        for key, child in node.children.items():
            if self._cleanup_recursive(child, cutoff_time, removed_count):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del node.children[key]
        return not node.children and not node.nodes and not node.is_end

    def get_stats(self) -> dict:
        with self._lock:
            return {
                'total_nodes': self._count_nodes(self._root),
                'max_depth': self._max_depth,
            }

    def _count_nodes(self, node: TrieNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


class PrefixCacheStrategy(BaseRoutingStrategy):
    """Prefix cache aware routing strategy.

    Routes requests to backend nodes based on prompt prefix matching.
    Requests with shared prefixes are routed to the same node to maximize
    KV cache reuse, similar to vLLM-router's implementation.
    """

    def select_node(
        self,
        model_name: str,
        candidates: dict,
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Select a node based on prefix cache matching."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            logger.warning(f'No candidates for model {model_name}')
            return None
        candidate_urls = list(matched.keys())
        if not request_key:
            logger.debug(f'No request_key, fallback to first candidate: {candidate_urls[0]}')
            return candidate_urls[0]

        prompt_preview = request_key[:50] + '...' if len(request_key) > 50 else request_key
        logger.info(f'Prefix cache lookup: prompt="{prompt_preview}", candidates={candidate_urls}')

        best_node = self._cache.find_best_node(request_key, candidate_urls)
        if best_node:
            logger.info(f'Prefix cache HIT: routing to {best_node}')
            self._cache.add_prefix(request_key, best_node)
            return best_node

        # Cache miss: select node with minimum unfinished requests (load balancing)
        selected = self._select_least_loaded_node(matched)
        logger.info(f'Prefix cache MISS: routing to {selected} (least loaded), caching prefix')
        self._cache.add_prefix(request_key, selected)
        return selected

    def __init__(self, max_prefix_depth: int = 100) -> None:
        self._cache = PrefixCacheTrie(max_depth=max_prefix_depth)
        self._max_depth = max_prefix_depth
        self._rr_counter = 0  # Round-robin counter for load balancing

    def _select_least_loaded_node(self, candidates: dict) -> str:
        """Select the node with minimum unfinished requests.
        
        Uses round-robin as tie-breaker when all nodes have same load.
        
        Args:
            candidates: Dict of node_url -> NodeStatus
            
        Returns:
            Selected node URL with minimum load.
        """
        if not candidates:
            raise ValueError("No candidates available")
        
        candidate_list = list(candidates.items())
        
        # Find minimum unfinished count
        min_unfinished = min(status.unfinished for _, status in candidate_list)
        
        # Get all nodes with minimum unfinished count
        min_load_nodes = [
            (url, status) for url, status in candidate_list 
            if status.unfinished == min_unfinished
        ]
        
        # If multiple nodes have same minimum load, use round-robin
        if len(min_load_nodes) > 1:
            self._rr_counter += 1
            selected_idx = self._rr_counter % len(min_load_nodes)
            selected_url, selected_status = min_load_nodes[selected_idx]
            logger.info(f'Load balancing: {len(min_load_nodes)} nodes with same load ({min_unfinished}), '
                       f'using round-robin index {selected_idx}')
        else:
            selected_url, selected_status = min_load_nodes[0]
        
        # Log all nodes' load info
        load_info = [
            f"{url}(unfinished={status.unfinished}, latency={sum(status.latency)/len(status.latency) if status.latency else 0:.3f}s)"
            for url, status in candidate_list
        ]
        logger.info(f'Load balancing candidates: {load_info}')
        logger.info(f'Selected: {selected_url} (unfinished={selected_status.unfinished})')
        
        return selected_url

    def update_cache(self, prompt: str, node_url: str) -> None:
        self._cache.add_prefix(prompt, node_url)

    def remove_node(self, node_url: str) -> None:
        self._cache.remove_node(node_url)

    def cleanup(self, max_age_seconds: float = 3600) -> int:
        return self._cache.cleanup_expired(max_age_seconds)

    def get_stats(self) -> dict:
        return self._cache.get_stats()
