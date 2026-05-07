"""SGLang backend adapter."""

from typing import TYPE_CHECKING, Any, Optional

import aiohttp
import requests

from dlrouter.backends.base import BaseBackend, CLIArg, PDRequestContext
from dlrouter.backends.http import BackendHTTPTransportMixin, StreamFraming
from dlrouter.backends.sglang.config import SGLangPDConfig
from dlrouter.backends.utils import normalize_backend_url, parse_csv_list
from dlrouter.constants import (
    AIOHTTP_TIMEOUT,
    HEALTH_CHECK_TIMEOUT,
    EngineRole,
    ServiceDiscoveryMode,
)
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.backends.pd import DualDispatchExecutor
    from dlrouter.core.node_manager import NodeManager
    from dlrouter.core.service_discovery.base import BaseServiceDiscovery


logger = get_logger('dlrouter.backends.sglang')

DEFAULT_BOOTSTRAP_PORT = 8998


class SGLangBackend(BackendHTTPTransportMixin, BaseBackend):
    """Backend adapter for SGLang inference engine."""

    stream_framing = StreamFraming.PASSTHROUGH

    def __init__(self, pd_config: Optional[SGLangPDConfig] = None) -> None:
        self.pd_config = pd_config or SGLangPDConfig()
        self._timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        self._health_timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
        self._connector_kwargs = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = None
        self._dual_dispatch_executor: Optional[DualDispatchExecutor] = None

    @classmethod
    def create(cls, parsed_config: Any = None) -> 'SGLangBackend':
        """Create a SGLang backend instance from parsed configuration."""
        config = parsed_config if isinstance(parsed_config, SGLangPDConfig) else SGLangPDConfig()
        return cls(pd_config=config)

    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available models from a SGLang node."""
        try:
            resp = requests.get(
                f'{node_url}/v1/models',
                headers={'accept': 'application/json'},
            )
            resp.raise_for_status()
            data = resp.json()
            return [m['id'] for m in data.get('data', [])]
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    def deregister_node(self, node_url: str) -> None:
        """No-op for SGLang static HTTP nodes."""

    def supports_pd_disagg(self) -> bool:
        """SGLang backend supports PD disaggregation."""
        return True

    def preferred_discovery_mode(
        self,
        backend_config: dict[str, Any],
    ) -> ServiceDiscoveryMode:
        """Return the SGLang discovery mode inferred from parsed PD config."""
        return self.parse_config(**backend_config).discovery_mode

    @classmethod
    def get_cli_args(cls) -> list[CLIArg]:
        """Return SGLang-specific CLI arguments."""
        return [
            CLIArg(
                name='prefill_urls',
                type=str,
                default=None,
                help='Comma-separated SGLang prefill URLs (static mode)',
            ),
            CLIArg(
                name='decode_urls',
                type=str,
                default=None,
                help='Comma-separated SGLang decode URLs (static mode)',
            ),
            CLIArg(
                name='prefill_bootstrap_ports',
                type=str,
                default=None,
                help='Comma-separated bootstrap ports aligned with prefill_urls; defaults to 8998',
            ),
            CLIArg(
                name='models',
                type=str,
                default=None,
                help='Comma-separated model names for SGLang PD mode',
            ),
        ]

    @classmethod
    def parse_config(cls, **kwargs: Any) -> SGLangPDConfig:
        """Parse SGLang-specific config from CLI args."""
        models = parse_csv_list(kwargs.get('models'))
        prefill_urls = parse_csv_list(kwargs.get('prefill_urls'))
        decode_urls = parse_csv_list(kwargs.get('decode_urls'))

        if bool(prefill_urls) != bool(decode_urls):
            raise ValueError('prefill_urls and decode_urls must be provided together')

        port_values = parse_csv_list(kwargs.get('prefill_bootstrap_ports'))
        if port_values:
            if len(port_values) != len(prefill_urls):
                raise ValueError('prefill_bootstrap_ports must match prefill_urls length')
            prefill_bootstrap_ports = [int(port) for port in port_values]
        else:
            prefill_bootstrap_ports = [DEFAULT_BOOTSTRAP_PORT for _ in prefill_urls]

        return SGLangPDConfig(
            discovery_mode=ServiceDiscoveryMode.STATIC,
            models=models,
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
            prefill_bootstrap_ports=prefill_bootstrap_ports,
        )

    def create_service_discovery(
        self,
        discovery_mode: ServiceDiscoveryMode,
        backend_config: dict[str, Any],
        node_manager: 'NodeManager',
    ) -> 'BaseServiceDiscovery':
        """Create static service discovery for SGLang PD mode."""
        from dlrouter.core.service_discovery import (
            NodeInfo,
            StaticServiceDiscovery,
        )

        config = self.parse_config(**backend_config)
        if discovery_mode != ServiceDiscoveryMode.STATIC:
            raise ValueError('SGLang backend currently supports only static discovery')
        if not config.prefill_urls or not config.decode_urls:
            raise ValueError('SGLang backend currently requires static prefill_urls and decode_urls')

        prefill_instances = [
            NodeInfo(
                http_address=normalize_backend_url(url, strip_scheme=True),
                role=EngineRole.PREFILL,
                models=config.models,
            )
            for url in config.prefill_urls
        ]
        decode_instances = [
            NodeInfo(
                http_address=normalize_backend_url(url, strip_scheme=True),
                role=EngineRole.DECODE,
                models=config.models,
            )
            for url in config.decode_urls
        ]

        return StaticServiceDiscovery(
            node_manager=node_manager,
            models=config.models,
            prefill_instances=prefill_instances,
            decode_instances=decode_instances,
        )

    async def handle_pd_request(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Handle request in SGLang PD disaggregation mode."""
        return await self._get_dual_dispatch_executor().execute(
            request_data=request_data,
            endpoint=endpoint,
            stream=stream,
            context=context,
        )

    def _get_dual_dispatch_executor(self) -> 'DualDispatchExecutor':
        """Return the cached SGLang dual-dispatch executor."""
        if self._dual_dispatch_executor is None:
            self._dual_dispatch_executor = self._build_dual_dispatch_executor()
        return self._dual_dispatch_executor

    def _build_dual_dispatch_executor(self) -> 'DualDispatchExecutor':
        """Build the configured SGLang dual-dispatch executor."""
        from dlrouter.backends.pd import DualDispatchExecutor
        from dlrouter.backends.sglang.bootstrap import SGLangBootstrapAdapter

        port_map: dict[str, Optional[int]] = dict(
            zip(
                self.pd_config.prefill_urls,
                self.pd_config.prefill_bootstrap_ports,
            )
        )
        return DualDispatchExecutor(
            transport=self,
            adapter=SGLangBootstrapAdapter(port_map),
        )
