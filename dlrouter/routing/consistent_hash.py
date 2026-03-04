"""Consistent hash routing strategy."""

import bisect
import hashlib
from typing import Optional

from dlrouter.models.node import NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy


# Number of virtual nodes per real node
DEFAULT_REPLICAS = 150


class ConsistentHashRing:
    """A consistent hash ring implementation.

    Maps keys to nodes using virtual nodes for
    balanced distribution.
    """

    def __init__(self, replicas: int = DEFAULT_REPLICAS):
        self.replicas = replicas
        self._ring: list[tuple[int, str]] = []
        self._sorted_keys: list[int] = []

    def _hash(self, key: str) -> int:
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest, 16)

    def build(self, nodes: list[str]) -> None:
        """Build the hash ring from node list."""
        self._ring.clear()
        self._sorted_keys.clear()
        for node in nodes:
            for i in range(self.replicas):
                vkey = f'{node}#{i}'
                h = self._hash(vkey)
                self._ring.append((h, node))
        self._ring.sort(key=lambda x: x[0])
        self._sorted_keys = [h for h, _ in self._ring]

    def get_node(self, key: str) -> Optional[str]:
        """Get the node for a given key."""
        if not self._ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_keys, h)
        if idx == len(self._sorted_keys):
            idx = 0
        return self._ring[idx][1]


class ConsistentHashStrategy(BaseRoutingStrategy):
    """Consistent hash routing.

    Routes requests to the same node based on a
    request key (e.g., user id, session id, or
    the hash of the first message).
    """

    def __init__(self, replicas: int = DEFAULT_REPLICAS) -> None:
        self._replicas = replicas
        self._ring = ConsistentHashRing(replicas)

    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Select node by consistent hashing."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            return None

        urls = sorted(matched.keys())
        self._ring.build(urls)

        if request_key is None:
            request_key = model_name
        return self._ring.get_node(request_key)
