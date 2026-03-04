"""Random / weighted-random routing strategy."""

import random as _random
from typing import Optional

from dlrouter.models.node import NodeStatus
from dlrouter.routing.base import BaseRoutingStrategy


class RandomStrategy(BaseRoutingStrategy):
    """Weighted random routing.

    Nodes with a ``speed`` attribute get proportionally
    more traffic.  Nodes without speed are assigned the
    average speed of nodes that do report it.
    """

    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Select a random node weighted by speed."""
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

        total = sum(all_speeds)
        weights = [s / total for s in all_speeds]
        idx = _random.choices(range(len(all_urls)), weights=weights)[0]
        return all_urls[idx]
