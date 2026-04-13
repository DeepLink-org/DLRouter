"""vLLM backend adapter.

Supports standard OpenAI-compatible API forwarding
for vLLM inference engine, including PD disaggregation mode.
"""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Optional

import aiohttp
import requests
from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.base import BaseBackend, CLIArg, PDRequestContext
from dlrouter.backends.vllm.config import VLLMPDConfig
from dlrouter.constants import (
    AIOHTTP_TIMEOUT,
    HEALTH_CHECK_TIMEOUT,
    EngineRole,
    ServiceDiscoveryMode,
)
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager
    from dlrouter.core.service_discovery.base import BaseServiceDiscovery


logger = get_logger('dlrouter.backends.vllm')

DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100


class VLLMBackend(BaseBackend):
    """Backend adapter for vLLM inference engine."""

    def __init__(
        self,
        pool_connections: int = DEFAULT_POOL_CONNECTIONS,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    ) -> None:
        self._timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        self._health_timeout = aiohttp.ClientTimeout(
            total=HEALTH_CHECK_TIMEOUT,
        )
        self._connector_kwargs = {
            'limit': pool_connections,
            'limit_per_host': pool_maxsize,
            'ttl_dns_cache': 300,
            'enable_cleanup_closed': True,
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a persistent aiohttp session."""
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()

        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(**self._connector_kwargs)
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=self._timeout,
                    )
        return self._session

    async def close(self) -> None:
        """Close the persistent session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward request to vLLM node."""
        session = await self._get_session()
        url = node_url + endpoint
        try:
            async with session.post(
                url,
                json=request_data,
                timeout=self._timeout,
            ) as resp:
                return await resp.text()
        except Exception as e:
            logger.error(f'Forward error: {e}')
            raise

    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream-forward request to vLLM node."""
        session = await self._get_session()
        url = node_url + endpoint
        try:
            async with session.post(
                url,
                json=request_data,
                timeout=self._timeout,
            ) as resp:
                async for line in resp.content:
                    if line.strip():
                        yield line + b'\n\n'
        except Exception as e:
            logger.error(f'Stream error: {e}')
            raise

    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available models from vLLM node."""
        try:
            url = f'{node_url}/v1/models'
            headers = {'accept': 'application/json'}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return [m['id'] for m in data.get('data', [])]
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    async def check_health(self, node_url: str) -> bool:
        """Check vLLM node health via async request."""
        url = f'{node_url}/health'
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    url,
                    timeout=self._health_timeout,
                ) as resp,
            ):
                return resp.status == 200
        except Exception as e:
            logger.error(f'Failed to check health from {node_url}: {e}')
            return False

    def deregister_node(self, node_url: str) -> None:
        """No-op for vLLM (no PD connection pool)."""

    async def forward_with_request_id(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ) -> Any:
        """Forward request to vLLM node with X-Request-Id header."""
        session = await self._get_session()
        url = node_url + endpoint
        headers = {
            'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY", "")}',
            'X-Request-Id': request_id,
        }
        try:
            async with session.post(
                url,
                json=request_data,
                headers=headers,
                timeout=self._timeout,
            ) as resp:
                return await resp.text()
        except Exception as e:
            logger.error(f'Forward with request_id error: {e}')
            raise

    async def stream_forward_with_request_id(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """Stream-forward request to vLLM node with X-Request-Id header."""
        session = await self._get_session()
        url = node_url + endpoint
        headers = {
            'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY", "")}',
            'X-Request-Id': request_id,
        }
        try:
            async with session.post(
                url,
                json=request_data,
                headers=headers,
                timeout=self._timeout,
            ) as resp:
                async for chunk in resp.content.iter_chunked(1024):
                    yield chunk
        except Exception as e:
            logger.error(f'Stream with request_id error: {e}')
            raise

    @classmethod
    def get_cli_args(cls) -> list[CLIArg]:
        """Return vLLM-specific CLI arguments."""
        return [
            CLIArg(
                name='discovery_mode',
                type=str,
                default='heartbeat',
                help='Service discovery mode: static (manual config) or heartbeat (ZMQ)',
                choices=['static', 'heartbeat'],
            ),
            CLIArg(
                name='zmq_host',
                type=str,
                default='0.0.0.0',
                help='ZMQ service discovery bind host (heartbeat mode)',
            ),
            CLIArg(
                name='zmq_port',
                type=int,
                default=30001,
                help='ZMQ service discovery port (heartbeat mode)',
            ),
            CLIArg(
                name='zmq_ping_timeout',
                type=int,
                default=5,
                help='ZMQ ping timeout in seconds (heartbeat mode)',
            ),
            CLIArg(
                name='prefill_urls',
                type=str,
                default=None,
                help='Comma-separated prefill URLs (static mode)',
            ),
            CLIArg(
                name='decode_urls',
                type=str,
                default=None,
                help='Comma-separated decode URLs (static mode)',
            ),
            CLIArg(
                name='models',
                type=str,
                default=None,
                help='Comma-separated model names for vLLM PD mode',
            ),
        ]

    @classmethod
    def parse_config(cls, **kwargs) -> VLLMPDConfig:
        """Parse vLLM-specific config from CLI args."""
        models = []
        if kwargs.get('models'):
            models = [m.strip() for m in kwargs['models'].split(',')]

        prefill_urls = []
        if kwargs.get('prefill_urls'):
            prefill_urls = [u.strip() for u in kwargs['prefill_urls'].split(',')]

        decode_urls = []
        if kwargs.get('decode_urls'):
            decode_urls = [u.strip() for u in kwargs['decode_urls'].split(',')]

        discovery_mode = ServiceDiscoveryMode(kwargs.get('discovery_mode', 'heartbeat'))

        return VLLMPDConfig(
            discovery_mode=discovery_mode,
            zmq_host=kwargs.get('zmq_host', '0.0.0.0'),
            zmq_port=kwargs.get('zmq_port', 30001),
            ping_timeout_seconds=kwargs.get('zmq_ping_timeout', 5),
            models=models,
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
        )

    def create_service_discovery(
        self,
        discovery_mode: ServiceDiscoveryMode,
        backend_config: dict[str, Any],
        node_manager: 'NodeManager',
    ) -> 'BaseServiceDiscovery':
        """Create service discovery for vLLM PD mode."""
        from dlrouter.core.service_discovery import (
            NodeInfo,
            StaticServiceDiscovery,
            ZMQHeartbeatDiscovery,
        )

        config = self.parse_config(**backend_config)

        if discovery_mode == ServiceDiscoveryMode.STATIC:
            prefill_instances = [
                NodeInfo(
                    http_address=url.replace('http://', '').replace('https://', ''),
                    role=EngineRole.PREFILL,
                    models=config.models,
                )
                for url in config.prefill_urls
            ]
            decode_instances = [
                NodeInfo(
                    http_address=url.replace('http://', '').replace('https://', ''),
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

        if discovery_mode == ServiceDiscoveryMode.HEARTBEAT:
            return ZMQHeartbeatDiscovery(
                host=config.zmq_host,
                port=config.zmq_port,
                ping_timeout_seconds=config.ping_timeout_seconds,
                node_manager=node_manager,
                models=config.models,
            )

        raise ValueError(f'Unknown discovery mode: {discovery_mode}')

    async def handle_pd_request(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Handle request in vLLM PD disaggregation mode."""
        service_discovery = context.service_discovery
        if service_discovery is None:
            logger.warning('No service discovery configured for vLLM PD request')
            return JSONResponse(
                {'error': 'No service discovery configured for vLLM PD mode'},
                status_code=503,
            )

        pd_pair = service_discovery.select_pd_pair()
        if pd_pair is None:
            logger.warning('No P/D instances available')
            return JSONResponse(
                {'error': 'No prefill or decode instances available'},
                status_code=503,
            )

        prefill_info, decode_info = pd_pair
        base_id = uuid.uuid4().hex
        request_id = service_discovery.build_request_id(prefill_info, decode_info, base_id)

        logger.info(
            f'vLLM PD: [HTTP:{prefill_info.http_address}, ZMQ:{prefill_info.zmq_address}] -> '
            f'[HTTP:{decode_info.http_address}, ZMQ:{decode_info.zmq_address}]'
        )

        p_url = prefill_info.to_http_url()
        d_url = decode_info.to_http_url()

        prefill_request = request_data.copy()
        prefill_request['max_tokens'] = 1
        if 'max_completion_tokens' in prefill_request:
            prefill_request['max_completion_tokens'] = 1

        try:
            async for _ in self.stream_forward_with_request_id(p_url, endpoint, prefill_request, request_id):
                pass
        except Exception as e:
            logger.error(f'vLLM PD prefill error: {e}')
            return JSONResponse(
                {'error': f'Prefill phase failed: {e}'},
                status_code=502,
            )

        if stream:
            gen = self._stream_generate_pd(d_url, endpoint, request_data, request_id)
            return StreamingResponse(
                gen,
                media_type='text/event-stream',
            )

        try:
            text = await self.forward_with_request_id(d_url, endpoint, request_data, request_id)
            return JSONResponse(json.loads(text))
        except Exception as e:
            logger.error(f'vLLM PD decode error: {e}')
            return JSONResponse(
                {'error': f'Decode phase failed: {e}'},
                status_code=502,
            )

    async def _stream_generate_pd(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """Async generator for vLLM PD streaming with request_id."""
        try:
            gen = self.stream_forward_with_request_id(node_url, endpoint, request_data, request_id)
            async for chunk in gen:
                yield chunk
        except Exception as e:
            logger.error(f'vLLM PD stream error: {e}')
            error_data = {'error': f'Stream failed: {e}'}
            yield json.dumps(error_data).encode() + b'\n'
