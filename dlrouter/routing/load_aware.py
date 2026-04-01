"""Load-aware routing strategies."""

import random as _random
from typing import Optional

from dlrouter.models.node import NodeMetrics, NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy


class MinExpectedLatencyStrategy(BaseRoutingStrategy):
    """Select the node with minimum expected latency.

    Expected latency = unfinished_requests / speed.
    Uses NodeMetrics with __slots__ for optimized routing calculations.
    """

    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Pick node with lowest expected latency."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            return None

        # Convert to lightweight metrics objects
        metrics_list: list[NodeMetrics] = [NodeMetrics.from_status(url, st) for url, st in matched.items()]

        # Calculate average speed for nodes without speed data
        speeds = [m.speed for m in metrics_list if m.speed is not None]
        avg_speed = sum(speeds) / len(speeds) if speeds else 1.0

        # Shuffle to break ties randomly
        _random.shuffle(metrics_list)

        best_url = metrics_list[0].url
        best_lat = float('inf')
        for m in metrics_list:
            spd = m.speed if m.speed is not None else avg_speed
            if spd <= 0:
                spd = 1e-6
            lat = m.unfinished / spd
            if lat < best_lat:
                best_lat = lat
                best_url = m.url

        return best_url


class MinObservedLatencyStrategy(BaseRoutingStrategy):
    """Select the node with minimum expected waiting time.

    Expected waiting time = unfinished_requests * avg_observed_latency.
    Uses NodeMetrics with __slots__ for optimized routing calculations.

    Cold start handling:
    - If all nodes have no data: use default latency 1.0
    - If some nodes have data: cold nodes use average latency of warm nodes
    - This ensures fair comparison between cold and warm nodes
    """

    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Pick node with lowest expected waiting time."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            return None

        # Convert to lightweight metrics objects
        metrics_list: list[NodeMetrics] = [NodeMetrics.from_status(url, st) for url, st in matched.items()]

        # Calculate average latency for nodes with data
        latencies = [m.avg_latency for m in metrics_list if m.avg_latency]
        avg_latency = sum(latencies) / len(latencies) if latencies else 1.0

        # Calculate expected waiting time for each node
        wait_times: list[tuple[str, float]] = []
        for m in metrics_list:
            lat = m.avg_latency if m.avg_latency else avg_latency
            wait = m.unfinished * lat
            wait_times.append((m.url, wait))

        # Find minimum wait time
        min_wait = min(w for _, w in wait_times)

        # Select randomly among best candidates (tolerance for float)
        best = [url for url, wait in wait_times if wait <= min_wait + 1e-9]
        return _random.choice(best)
