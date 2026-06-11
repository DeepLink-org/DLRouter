"""dlslime-ctrl based service discovery for DLEngine HTTP servers.

DLEngine ``serve`` registers OpenAI-compatible HTTP endpoints with
dlslime-ctrl (entity kind ``dlengine``). This discovery polls
``list_entities`` and syncs them into NodeManager.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery.base import BaseServiceDiscovery
from dlrouter.logger import get_logger
from dlrouter.models.node import NodeStatus


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.service_discovery.nanoctrl')

DEFAULT_CTRL_KIND = 'dlengine'


def _entity_http_url(entity: dict[str, Any]) -> str | None:
    """Build node URL from a dlslime-ctrl entity record."""
    endpoint = entity.get('endpoint') or {}
    if not isinstance(endpoint, dict):
        return None
    host = endpoint.get('host')
    port = endpoint.get('port')
    if not host or port is None:
        return None
    protocol = endpoint.get('protocol', 'http')
    if protocol == 'https':
        return f'https://{host}:{port}'
    return f'http://{host}:{port}'


_ROLE_BY_NAME = {
    'prefill': EngineRole.PREFILL,
    'decode': EngineRole.DECODE,
    'hybrid': EngineRole.HYBRID,
}


def _entity_role(entity: dict[str, Any]) -> EngineRole:
    """Map the entity ``metadata.role`` to an EngineRole (default HYBRID)."""
    metadata = entity.get('metadata') or {}
    if not isinstance(metadata, dict):
        return EngineRole.HYBRID
    role = str(metadata.get('role', 'hybrid')).strip().lower()
    return _ROLE_BY_NAME.get(role, EngineRole.HYBRID)


def _entity_models(entity: dict[str, Any]) -> list[str]:
    """Model aliases for routing (served name, path, basename)."""
    metadata = entity.get('metadata') or {}
    if not isinstance(metadata, dict):
        return []

    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str | None) -> None:
        if not name:
            return
        key = name.strip()
        if not key or key in seen:
            return
        seen.add(key)
        names.append(key)

    _add(metadata.get('served_model_name'))
    model_path = metadata.get('model_path')
    if model_path:
        path = str(model_path).rstrip('/')
        _add(path)
        _add(path.split('/')[-1])
    return names


class NanoCtrlServiceDiscovery(BaseServiceDiscovery):
    """Poll dlslime-ctrl and reconcile DLEngine HTTP nodes."""

    def __init__(
        self,
        ctrl_address: str,
        node_manager: Optional[NodeManager] = None,
        ctrl_scope: Optional[str] = None,
        ctrl_kind: str = DEFAULT_CTRL_KIND,
        poll_interval: float = 5.0,
    ) -> None:
        super().__init__(node_manager=node_manager)
        self._ctrl_address = ctrl_address
        self._ctrl_scope = ctrl_scope
        self._ctrl_kind = ctrl_kind
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._known_urls: set[str] = set()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from dlslime.ctrl import NanoCtrlClient
            except ImportError as e:
                raise ImportError(
                    'dlslime is required for DLEngine dlslime-ctrl discovery. '
                    'Install with: pip install dlslime',
                ) from e
            self._client = NanoCtrlClient(self._ctrl_address, self._ctrl_scope)
            self._client.check_connection()
        return self._client

    def _poll_once(self) -> None:
        client = self._get_client()
        entities = client.list_entities(kind=self._ctrl_kind)
        live_urls: set[str] = set()

        for entity in entities:
            node_url = _entity_http_url(entity)
            if not node_url:
                logger.warning(f'Skipping entity without HTTP endpoint: {entity}')
                continue
            live_urls.add(node_url)
            if node_url in self._known_urls:
                continue
            models = _entity_models(entity)
            if self._node_manager is None:
                self._known_urls.add(node_url)
                continue
            role = _entity_role(entity)
            status = NodeStatus(role=role, models=models)
            if self._node_manager.add(node_url, status):
                logger.info(
                    f'Discovered DLEngine node {node_url} '
                    f'role={role.name} models={models}',
                )
            self._known_urls.add(node_url)

        stale = self._known_urls - live_urls
        for node_url in stale:
            self._known_urls.discard(node_url)
            if self._node_manager is not None:
                self._node_manager.remove(node_url)
                logger.info(f'Removed stale DLEngine node {node_url}')

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_interval):
            try:
                self._poll_once()
            except Exception as e:
                logger.error(f'NanoCtrl discovery poll failed: {e}', exc_info=True)

    def start(self) -> None:
        self._running = True
        try:
            self._poll_once()
        except Exception as e:
            logger.error(f'NanoCtrl discovery initial poll failed: {e}')
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name='dlrouter-nanoctrl-discovery',
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f'NanoCtrl discovery started (ctrl={self._ctrl_address}, '
            f'kind={self._ctrl_kind}, interval={self._poll_interval}s)',
        )

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval + 2.0)
        self._thread = None
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
        logger.info('NanoCtrl discovery stopped')

    def unregister_by_url(self, node_url: str) -> None:
        """Allow HealthChecker removals to re-discover the same URL later."""
        self._known_urls.discard(node_url)
