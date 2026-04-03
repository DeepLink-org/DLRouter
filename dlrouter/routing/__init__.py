"""Routing strategies for DLRouter."""

from dlrouter.routing.base import BaseRoutingStrategy
from dlrouter.routing.consistent_hash import (
    ConsistentHashStrategy,
)
from dlrouter.routing.factory import create_routing_strategy
from dlrouter.routing.load_aware import (
    MinExpectedLatencyStrategy,
    MinObservedLatencyStrategy,
)
from dlrouter.routing.prefix_cache import PrefixCacheStrategy
from dlrouter.routing.random_strategy import RandomStrategy
from dlrouter.routing.round_robin import RoundRobinStrategy


__all__ = [
    'BaseRoutingStrategy',
    'ConsistentHashStrategy',
    'MinExpectedLatencyStrategy',
    'MinObservedLatencyStrategy',
    'PrefixCacheStrategy',
    'RandomStrategy',
    'RoundRobinStrategy',
    'create_routing_strategy',
]
