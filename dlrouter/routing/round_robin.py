"""Round-robin routing strategy."""

import threading
from typing import Optional

from dlrouter.models.node import NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy


class RoundRobinStrategy(BaseRoutingStrategy):
    """Round-robin routing across available nodes."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Select next node in round-robin order."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            return None

        urls = sorted(matched.keys())
        with self._lock:
            idx = self._counter % len(urls)
            self._counter += 1
        return urls[idx]
