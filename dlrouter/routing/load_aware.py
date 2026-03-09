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
    """Select the node with minimum observed latency.

    Based on average latency of recent requests.

    Note: To avoid "winner-takes-all" problem in multi-node scenarios,
    when multiple nodes have similar latencies (within 50% of the minimum),
    randomly select among them to achieve load balancing.

    Cold start handling:
    - Nodes with no latency data are assigned a default latency of 1.0
    - This allows unified latency-based scheduling without special cases
    """

    DEFAULT_LATENCY = 1.0  # Default latency for cold start nodes

    def _get_avg_latency(self, st: NodeStatus) -> float:
        """Get average latency from node status, default if no data."""
        if not st.latency:
            return self.DEFAULT_LATENCY
        return float(np.mean(np.array(list(st.latency))))

    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Pick node with lowest observed latency."""
        matched = self._filter_by_model(model_name, candidates)
        if not matched:
            return None

        # Build latency map (use default for cold start)
        lat_map = {url: self._get_avg_latency(st) for url, st in matched.items()}
        min_lat = min(lat_map.values())

        # Find all nodes with latency within 50% of the minimum
        similar = [url for url, lat in lat_map.items() if lat < min_lat * 1.5]

        # Among similar latency nodes, select the one with minimum unfinished
        min_unfinished = min(matched[url].unfinished for url in similar)
        best = [url for url in similar if matched[url].unfinished == min_unfinished]
        selected = _random.choice(best)

        logger.debug(
            f'MinObservedLatency: min_lat={min_lat:.3f}, '
            f'similar={len(similar)}, best_unfinished={min_unfinished}, '
            f'candidates={len(best)}, selected={selected}'
        )
        return selected
