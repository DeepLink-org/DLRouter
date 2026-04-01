"""Node and status models."""

from collections import deque
from typing import Optional

from pydantic import BaseModel, Field

from dlrouter.constants import LATENCY_DEQUE_LEN, EngineRole


class NodeMetrics:
    """Lightweight metrics snapshot for routing decisions.

    Uses __slots__ for memory efficiency and faster attribute access.
    This class is used during routing calculations to avoid
    repeated attribute lookups on the heavier NodeStatus model.
    """

    __slots__ = ('avg_latency', 'speed', 'unfinished', 'url')

    def __init__(
        self,
        url: str,
        unfinished: int = 0,
        speed: Optional[float] = None,
        avg_latency: Optional[float] = None,
    ) -> None:
        self.url = url
        self.unfinished = unfinished
        self.speed = speed
        self.avg_latency = avg_latency

    @classmethod
    def from_status(
        cls,
        url: str,
        status: 'NodeStatus',
    ) -> 'NodeMetrics':
        """Create NodeMetrics from NodeStatus."""
        avg_latency = None
        if status.latency:
            # Inline calculation to avoid numpy import overhead
            lat_list = list(status.latency)
            avg_latency = sum(lat_list) / len(lat_list)
        return cls(
            url=url,
            unfinished=status.unfinished,
            speed=status.speed,
            avg_latency=avg_latency,
        )


class NodeStatus(BaseModel):
    """Status of a backend inference node."""

    model_config = {'arbitrary_types_allowed': True}

    role: EngineRole = EngineRole.HYBRID
    models: list[str] = Field(default_factory=list)
    unfinished: int = 0
    latency: deque = Field(
        default_factory=lambda: deque(maxlen=LATENCY_DEQUE_LEN),
    )
    speed: Optional[float] = None


class Node(BaseModel):
    """A backend node with url and optional status."""

    url: str
    status: Optional[NodeStatus] = None
