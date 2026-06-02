"""NanoDeploy backend adapter.

Forwards OpenAI-compatible HTTP to NanoDeploy ``serve`` nodes. When
``--ctrl_address`` is set, discovers nodes via dlslime-ctrl (entity kind
``nanodeploy``).
"""

from typing import TYPE_CHECKING, Any, Optional

import aiohttp
import requests

from dlrouter.backends.base import BaseBackend, CLIArg
from dlrouter.backends.http import BackendHTTPTransportMixin, StreamFraming
from dlrouter.backends.nanodeploy.config import NanoDeployConfig
from dlrouter.constants import (
    AIOHTTP_TIMEOUT,
    HEALTH_CHECK_TIMEOUT,
    ServiceDiscoveryMode,
)
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager
    from dlrouter.core.service_discovery.base import BaseServiceDiscovery


logger = get_logger('dlrouter.backends.nanodeploy')

DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100

# DLRouter adds routing metadata; NanoDeploy serve only needs generation fields.
_CHAT_FORWARD_KEYS = frozenset(
    {
        'model',
        'messages',
        'prompt',
        'stream',
        'temperature',
        'max_tokens',
        'max_completion_tokens',
        'ignore_eos',
        'stop',
    }
)


def _sanitize_chat_payload(request_data: dict[str, Any]) -> dict[str, Any]:
    """Keep a minimal OpenAI payload for NanoDeploy serve."""
    payload = {k: request_data[k] for k in _CHAT_FORWARD_KEYS if k in request_data}
    if 'model' in payload:
        payload['model'] = str(payload['model'])
    return payload


class NanoDeployBackend(BackendHTTPTransportMixin, BaseBackend):
    """Backend adapter for NanoDeploy OpenAI HTTP servers."""

    stream_framing = StreamFraming.SSE_LINES

    def __init__(
        self,
        config: Optional[NanoDeployConfig] = None,
        pool_connections: int = DEFAULT_POOL_CONNECTIONS,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    ) -> None:
        self.config = config or NanoDeployConfig()
        self._timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        self._health_timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
        self._connector_kwargs = {
            'limit': pool_connections,
            'limit_per_host': pool_maxsize,
            'ttl_dns_cache': 300,
            'enable_cleanup_closed': True,
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = None

    @classmethod
    def create(cls, parsed_config: Any = None) -> 'NanoDeployBackend':
        """Create a NanoDeploy backend from parsed configuration."""
        config = (
            parsed_config
            if isinstance(parsed_config, NanoDeployConfig)
            else NanoDeployConfig()
        )
        return cls(config=config)

    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available models from a NanoDeploy node."""
        try:
            resp = requests.get(
                f'{node_url}/v1/models',
                headers={'accept': 'application/json'},
                timeout=HEALTH_CHECK_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m['id'] for m in data.get('data', [])]
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    def deregister_node(self, node_url: str) -> None:
        """No-op for NanoDeploy hybrid HTTP nodes."""

    def _prepare_payload(self, endpoint: str, request_data: dict[str, Any]) -> dict[str, Any]:
        if endpoint in ('/v1/chat/completions', '/v1/completions'):
            return _sanitize_chat_payload(request_data)
        return request_data

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        return await super().forward_request(
            node_url,
            endpoint,
            self._prepare_payload(endpoint, request_data),
            stream=stream,
        )

    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ):
        payload = self._prepare_payload(endpoint, request_data)
        async for chunk in super().stream_forward(node_url, endpoint, payload):
            yield chunk

    def supports_pd_disagg(self) -> bool:
        """NanoDeploy hybrid serve integration does not use router PD yet."""
        return False

    def preferred_discovery_mode(
        self,
        backend_config: dict[str, Any],
    ) -> Optional[ServiceDiscoveryMode]:
        """Use dlslime-ctrl polling when ``ctrl_address`` is configured."""
        cfg = self.parse_config(**backend_config)
        if cfg.ctrl_address:
            return ServiceDiscoveryMode.NANOCTRL
        return None

    @classmethod
    def get_cli_args(cls) -> list[CLIArg]:
        """Return NanoDeploy-specific CLI arguments."""
        return [
            CLIArg(
                name='ctrl_address',
                type=str,
                default=None,
                help='dlslime-ctrl address (host:port) for NanoDeploy node discovery',
            ),
            CLIArg(
                name='ctrl_scope',
                type=str,
                default=None,
                help='dlslime-ctrl scope for multi-tenant isolation',
            ),
            CLIArg(
                name='ctrl_kind',
                type=str,
                default='nanodeploy',
                help='Entity kind to list from dlslime-ctrl (default: nanodeploy)',
            ),
            CLIArg(
                name='discovery_poll_interval',
                type=float,
                default=5.0,
                help='Seconds between dlslime-ctrl discovery polls',
            ),
        ]

    @classmethod
    def parse_config(cls, **kwargs: Any) -> NanoDeployConfig:
        """Parse NanoDeploy config from CLI args."""
        ctrl_address = kwargs.get('ctrl_address')
        if ctrl_address is not None:
            ctrl_address = str(ctrl_address).strip() or None
        ctrl_scope = kwargs.get('ctrl_scope')
        if ctrl_scope is not None:
            ctrl_scope = str(ctrl_scope).strip() or None
        ctrl_kind = kwargs.get('ctrl_kind') or 'nanodeploy'
        interval = float(kwargs.get('discovery_poll_interval', 5.0))
        return NanoDeployConfig(
            ctrl_address=ctrl_address,
            ctrl_scope=ctrl_scope,
            ctrl_kind=str(ctrl_kind),
            discovery_poll_interval=interval,
        )

    def create_service_discovery(
        self,
        discovery_mode: ServiceDiscoveryMode,
        backend_config: dict[str, Any],
        node_manager: 'NodeManager',
    ) -> Optional['BaseServiceDiscovery']:
        """Create dlslime-ctrl polling discovery."""
        if discovery_mode != ServiceDiscoveryMode.NANOCTRL:
            return None
        cfg = self.parse_config(**backend_config)
        if not cfg.ctrl_address:
            logger.warning('NanoCtrl discovery requested but ctrl_address is empty')
            return None
        from dlrouter.core.service_discovery.nanoctrl_discovery import (
            NanoCtrlServiceDiscovery,
        )

        return NanoCtrlServiceDiscovery(
            ctrl_address=cfg.ctrl_address,
            node_manager=node_manager,
            ctrl_scope=cfg.ctrl_scope,
            ctrl_kind=cfg.ctrl_kind,
            poll_interval=cfg.discovery_poll_interval,
        )
