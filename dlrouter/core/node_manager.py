"""Node manager - central registry for backend nodes."""

import copy
import json
import os
import os.path as osp
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

import requests

from dlrouter.backends.base import BaseBackend
from dlrouter.constants import (
    LATENCY_DEQUE_LEN,
    EngineRole,
    RoutingStrategy,
    ServingStrategy,
)
from dlrouter.logger import get_logger
from dlrouter.models.node import NodeStatus
from dlrouter.routing.factory import create_routing_strategy


if TYPE_CHECKING:
    from dlrouter.routing.base import BaseRoutingStrategy


logger = get_logger('dlrouter.node_manager')


class NodeManager:
    """Central registry and manager for backend nodes.

    Manages node registration, removal, status tracking,
    and delegates routing to the configured strategy.

    Args:
        backend: The backend adapter instance.
        routing_strategy: The routing strategy enum.
        serving_strategy: The serving strategy enum.
        config_path: Path to persist node state.
        cache_status: Whether to cache status to file.
    """

    def __init__(
        self,
        backend: BaseBackend,
        routing_strategy: RoutingStrategy = (RoutingStrategy.MIN_EXPECTED_LATENCY),
        serving_strategy: ServingStrategy = (ServingStrategy.HYBRID),
        config_path: Optional[str] = None,
        cache_status: bool = True,
    ) -> None:
        self.backend = backend
        self.serving_strategy = serving_strategy
        self.cache_status = cache_status
        self.nodes: dict[str, NodeStatus] = {}
        self._lock = threading.RLock()  # 保护 nodes 的并发访问

        # Routing strategy
        self._routing_strategy_enum = routing_strategy
        self._router: BaseRoutingStrategy = create_routing_strategy(routing_strategy)

        # Config persistence path
        default_path = osp.join(
            osp.dirname(osp.realpath(__file__)),
            'router_config.json',
        )
        self.config_path = config_path or default_path

        # Load cached config
        if osp.exists(self.config_path) and self.cache_status:
            self._load_config()

    @property
    def routing_strategy(self) -> RoutingStrategy:
        """Current routing strategy."""
        return self._routing_strategy_enum

    @routing_strategy.setter
    def routing_strategy(self, value: RoutingStrategy) -> None:
        self._routing_strategy_enum = value
        self._router = create_routing_strategy(value)

    # -- Config persistence --

    def _load_config(self) -> None:
        """Load node config from file."""
        try:
            if os.path.getsize(self.config_path) > 0:
                logger.info(f'Loading config: {self.config_path}')
                with open(self.config_path) as f:
                    data = json.load(f)
                with self._lock:
                    for url, raw in data.items():
                        status = NodeStatus.model_validate_json(raw)
                        # Re-create deque with maxlen since Pydantic loses maxlen during deserialization
                        status.latency = deque(
                            list(status.latency)[-LATENCY_DEQUE_LEN:],
                            maxlen=LATENCY_DEQUE_LEN,
                        )
                        self.nodes[url] = status
        except Exception as e:
            logger.warning(f'Failed to load config: {e}')

    def _save_config(self) -> None:
        """Persist node config to file."""
        if not self.cache_status:
            return
        try:
            with self._lock:
                nodes = copy.deepcopy(self.nodes)
            for _, st in nodes.items():
                st.latency = deque(
                    list(st.latency)[-LATENCY_DEQUE_LEN:],
                    maxlen=LATENCY_DEQUE_LEN,
                )
            with open(self.config_path, 'w') as f:
                json.dump(
                    {url: st.model_dump_json() for url, st in nodes.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f'Failed to save config: {e}')

    # -- Node CRUD --

    def add(
        self,
        node_url: str,
        status: Optional[NodeStatus] = None,
    ) -> bool:
        """Register a backend node.

        Args:
            node_url: URL of the backend node.
            status: Optional initial status.

        Returns:
            True when the node was registered, False when backend registration failed.
        """
        with self._lock:
            if status is None:
                status = self.nodes.get(node_url, NodeStatus())

        if status.models:
            self.remove(node_url)
            with self._lock:
                self.nodes[node_url] = status
            self._save_config()
            return True

        try:
            status = self.backend.register_node(node_url, status)
        except Exception as e:
            logger.error(f'Failed to add node {node_url}: {e}')
            return False

        with self._lock:
            self.nodes[node_url] = status
        self._save_config()
        return True

    def remove(self, node_url: str) -> None:
        """Remove a backend node.

        Args:
            node_url: URL to remove.
        """
        with self._lock:
            if node_url not in self.nodes:
                return
            self.nodes.pop(node_url)
        self._save_config()
        self.backend.deregister_node(node_url)

    def terminate_node(self, node_url: str) -> bool:
        """Terminate and remove a node.

        Args:
            node_url: URL to terminate.

        Returns:
            True if terminated successfully.
        """
        with self._lock:
            if node_url not in self.nodes:
                logger.error(f'Node {node_url} not found.')
                return False
            self.nodes.pop(node_url)
        try:
            resp = requests.get(
                f'{node_url}/terminate',
                headers={'accept': 'application/json'},
            )
            if resp.status_code != 200:
                logger.error(f'Terminate {node_url} failed: {resp.status_code}')
                return False
        except Exception as e:
            logger.error(f'Terminate error {node_url}: {e}')
            return False
        self._save_config()
        return True

    def terminate_all(self) -> bool:
        """Terminate all registered nodes."""
        with self._lock:
            urls = list(self.nodes.keys())
        ok = True
        for url in urls:
            if not self.terminate_node(url):
                ok = False
        return ok

    # -- Query --

    def get_nodes(self, role: EngineRole) -> dict[str, NodeStatus]:
        """Get nodes filtered by role."""
        with self._lock:
            return {url: st for url, st in list(self.nodes.items()) if st.role == role}

    @property
    def hybrid_nodes(self) -> dict[str, NodeStatus]:
        """All Hybrid-role nodes."""
        return self.get_nodes(EngineRole.HYBRID)

    @property
    def prefill_nodes(self) -> dict[str, NodeStatus]:
        """All Prefill-role nodes."""
        return self.get_nodes(EngineRole.PREFILL)

    @property
    def decode_nodes(self) -> dict[str, NodeStatus]:
        """All Decode-role nodes."""
        return self.get_nodes(EngineRole.DECODE)

    @property
    def model_list(self) -> list[str]:
        """All available model names across nodes."""
        with self._lock:
            names = []
            for st in self.nodes.values():
                names.extend(st.models)
            return names

    @property
    def status(self) -> dict[str, NodeStatus]:
        """Current node status dict."""
        with self._lock:
            return copy.copy(self.nodes)

    # -- Routing --

    def get_node_url(
        self,
        model_name: str,
        role: EngineRole = EngineRole.HYBRID,
        request_key: Optional[str] = None,
    ) -> Optional[str]:
        """Select a node URL for the given request.

        Args:
            model_name: Requested model name.
            role: Engine role filter.
            request_key: Optional key for hash routing.

        Returns:
            Selected node URL, or None.
        """
        candidates = self.get_nodes(role)
        return self._router.select_node(model_name, candidates, request_key)

    # -- Request lifecycle tracking --

    def pre_call(self, node_url: str) -> float:
        """Mark a request as started on a node.

        Returns:
            Start timestamp.
        """
        with self._lock:
            self.nodes[node_url].unfinished += 1
        return time.time()

    def post_call(self, node_url: str, start: float) -> None:
        """Mark a request as finished on a node.

        Args:
            node_url: The node URL.
            start: The start timestamp.
        """
        with self._lock:
            if node_url in self.nodes:
                self.nodes[node_url].unfinished -= 1
                elapsed = time.time() - start
                self.nodes[node_url].latency.append(elapsed)
