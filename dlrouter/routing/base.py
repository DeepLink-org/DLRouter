"""Base routing strategy interface."""

from abc import ABC, abstractmethod
from typing import Optional

from dlrouter.models.node import NodeStatus


class BaseRoutingStrategy(ABC):
    """Abstract base class for routing strategies.

    All routing strategies must implement ``select_node``
    which picks a node URL from the candidate pool.
    """

    @abstractmethod
    def select_node(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Select a node to handle the request.

        Args:
            model_name: The model requested.
            candidates: Map of node_url -> NodeStatus for
                nodes that serve this model.
            request_key: Optional key for hash-based routing
                (e.g. user id or session id).

        Returns:
            The selected node URL, or None if no candidate.
        """

    def _filter_by_model(
        self,
        model_name: str,
        candidates: dict[str, NodeStatus],
    ) -> dict[str, NodeStatus]:
        """Filter candidates that serve the given model."""
        return {url: st for url, st in candidates.items() if model_name in st.models}
