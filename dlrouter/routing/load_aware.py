"""Load-aware routing strategies."""

import random as _random
from typing import Optional

import numpy as np

from dlrouter.logger import get_logger
from dlrouter.models.node import NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy


logger = get_logger('dlrouter.proxy_engine')


class MinExpectedLatencyStrategy(BaseRoutingStrategy):
    """Select the node with minimum expected latency.

    Expected latency = unfinished_requests / speed.
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

        urls_with_speed = []
        speeds = []
        urls_without_speed = []

        for url, st in matched.items():
            if st.speed is not None:
                urls_with_speed.append(url)
                speeds.append(st.speed)
            else:
                urls_without_speed.append(url)

        avg = sum(speeds) / len(speeds) if speeds else 1.0
        all_urls = urls_with_speed + urls_without_speed
        all_speeds = speeds + [avg] * len(urls_without_speed)

        # Shuffle to break ties randomly
        indices = list(range(len(all_urls)))
        _random.shuffle(indices)

        best_idx = indices[0]
        best_lat = float('inf')
        for i in indices:
            url = all_urls[i]
            spd = all_speeds[i]
            if spd <= 0:
                spd = 1e-6
            lat = matched[url].unfinished / spd
            if lat < best_lat:
                best_lat = lat
                best_idx = i

        return all_urls[best_idx]


class MinObservedLatencyStrategy(BaseRoutingStrategy):
    """Select the node with minimum expected waiting time.

    Expected waiting time = unfinished_requests * avg_observed_latency.

    This combines both queue depth and historical latency to estimate
    how long a new request would need to wait on each node.

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
        """Pick node with lowest expected waiting time (unfinished * latency)."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            return None

        # Calculate average latency for nodes with data
        latencies_with_data = []
        for st in matched.values():
            if st.latency:
                latencies_with_data.append(float(np.mean(np.array(list(st.latency)))))

        # If no nodes have data, use default latency for all
        # If some nodes have data, use average of existing data for cold start nodes
        avg_latency = (
            sum(latencies_with_data) / len(latencies_with_data)
            if latencies_with_data
            else 1.0  # Default when all nodes are cold
        )

        # Build latency map
        lat_map = {}
        for url, st in matched.items():
            if st.latency:
                lat_map[url] = float(np.mean(np.array(list(st.latency))))
            else:
                lat_map[url] = avg_latency  # Use average of existing data for cold nodes

        # Calculate expected waiting time: unfinished * latency
        wait_map = {}
        for url, lat in lat_map.items():
            wait_map[url] = matched[url].unfinished * lat

        min_wait = min(wait_map.values())

        # Find all nodes with minimum expected waiting time (allow small tolerance for float)
        best = [url for url, wait in wait_map.items() if wait <= min_wait + 1e-9]
        selected = _random.choice(best)
        return selected
