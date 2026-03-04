"""Routing strategy factory."""

from dlrouter.constants import RoutingStrategy
from dlrouter.routing.base import BaseRoutingStrategy
from dlrouter.routing.consistent_hash import (
    ConsistentHashStrategy,
)
from dlrouter.routing.load_aware import (
    MinExpectedLatencyStrategy,
    MinObservedLatencyStrategy,
)
from dlrouter.routing.random_strategy import RandomStrategy
from dlrouter.routing.round_robin import RoundRobinStrategy


def create_routing_strategy(
    strategy: RoutingStrategy,
) -> BaseRoutingStrategy:
    """Create a routing strategy instance.

    Args:
        strategy: The routing strategy enum value.

    Returns:
        An instance of the corresponding strategy.

    Raises:
        ValueError: If strategy is not supported.
    """
    mapping = {
        RoutingStrategy.ROUND_ROBIN: RoundRobinStrategy,
        RoutingStrategy.RANDOM: RandomStrategy,
        RoutingStrategy.CONSISTENT_HASH: (ConsistentHashStrategy),
        RoutingStrategy.MIN_EXPECTED_LATENCY: (MinExpectedLatencyStrategy),
        RoutingStrategy.MIN_OBSERVED_LATENCY: (MinObservedLatencyStrategy),
    }
    cls = mapping.get(strategy)
    if cls is None:
        raise ValueError(f'Unsupported routing strategy: {strategy}. Available: {list(mapping.keys())}')
    return cls()
