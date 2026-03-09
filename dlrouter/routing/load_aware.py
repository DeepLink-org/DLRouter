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
    - If all nodes have no data: select by minimum unfinished count
    - If some nodes have no data: prioritize nodes without data to warm them up
    - If all nodes have data: select among nodes with similar latency
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

        # Find minimum latency (excluding inf)
        valid_latencies = [lat for lat in latencies if lat != float('inf')]
        min_lat = min(valid_latencies) if valid_latencies else float('inf')

        # Cold start handling:
        # If all nodes have no latency data, use unfinished count to balance load
        if min_lat == float('inf'):
            # Find minimum unfinished count
            min_unfinished = min(matched[u].unfinished for u in matched)
            # Get all nodes with the same minimum unfinished count
            candidates = [u for u in matched if matched[u].unfinished == min_unfinished]
            # Randomly select one to prevent load concentration in high concurrency
            selected_url = _random.choice(candidates)
            logger.debug(
                f'MinObservedLatency (cold start): '
                f'selected={selected_url} (unfinished={matched[selected_url].unfinished}, '
                f'candidates={len(candidates)}/{len(matched)})'
            )
            return selected_url

        # Partial cold start handling:
        # Some nodes have latency data, some don't
        # Prioritize nodes without data, but prevent overload by considering load balance
        has_no_data = [urls[i] for i, lat in enumerate(latencies) if lat == float('inf')]
        has_data = [urls[i] for i, lat in enumerate(latencies) if lat != float('inf')]

        if has_no_data:
            # Check if no-data nodes are overloaded compared to nodes with data
            min_unfinished_no_data = min(matched[u].unfinished for u in has_no_data)
            min_unfinished_with_data = min(matched[u].unfinished for u in has_data) if has_data else float('inf')

            # Threshold: if no-data nodes have 2+ more unfinished requests than data nodes,
            # include all nodes to share the load
            if has_data and min_unfinished_no_data >= min_unfinished_with_data + 2:
                # Load balance: include both no-data and with-data nodes with similar load
                all_candidates = has_no_data + has_data
                selected_url = _random.choice(all_candidates)
                logger.debug(
                    f'MinObservedLatency (load balance): '
                    f'selected={selected_url} (no_data_min={min_unfinished_no_data}, '
                    f'data_min={min_unfinished_with_data}, all_candidates={len(all_candidates)})'
                )
                return selected_url

            # Otherwise, prioritize no-data nodes with minimum unfinished
            candidates = [u for u in has_no_data if matched[u].unfinished == min_unfinished_no_data]
            selected_url = _random.choice(candidates)
            logger.debug(
                f'MinObservedLatency (partial cold start): '
                f'selected={selected_url} (unfinished={matched[selected_url].unfinished}, '
                f'candidates={len(candidates)}/{len(has_no_data)} no_data_nodes)'
            )
            return selected_url

        # Load balancing enhancement:
        # Find all nodes with latency within 50% of the minimum
        candidates_with_similar_lat = [
            urls[i]
            for i, lat in enumerate(latencies)
            if lat < min_lat * 1.5  # 50% tolerance
        ]

        # Randomly select one from candidates with similar latency
        selected_url = _random.choice(candidates_with_similar_lat)

        logger.debug(
            f'MinObservedLatency: min_lat={min_lat:.3f}, '
            f'candidates={len(candidates_with_similar_lat)}/{len(urls)}, '
            f'selected={selected_url}'
        )

        return selected_url
