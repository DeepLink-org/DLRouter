"""Load-aware routing strategies."""

import random as _random
from typing import Optional

import numpy as np

from dlrouter.models.node import NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy


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
    """

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

        urls = []
        latencies = []
        for url, st in matched.items():
            urls.append(url)
            if len(st.latency):
                latencies.append(np.mean(np.array(list(st.latency))))
            else:
                latencies.append(float('inf'))

        idx = int(np.argmin(np.array(latencies)))
        return urls[idx]
