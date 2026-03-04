"""Node and status models."""

from collections import deque
from typing import Optional

from pydantic import BaseModel, Field

from dlrouter.constants import LATENCY_DEQUE_LEN, EngineRole


class NodeStatus(BaseModel):
    """Status of a backend inference node."""

    model_config = {'arbitrary_types_allowed': True}

    role: EngineRole = EngineRole.HYBRID
    models: list[str] = Field(default_factory=list)
    unfinished: int = 0
    latency: deque = Field(default_factory=lambda: deque(maxlen=LATENCY_DEQUE_LEN))
    speed: Optional[float] = None


class Node(BaseModel):
    """A backend node with url and optional status."""

    url: str
    status: Optional[NodeStatus] = None
