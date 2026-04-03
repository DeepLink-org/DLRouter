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
from pydantic import BaseModel, Field

from dlrouter.backends.base import BaseBackend, CLIArg
from dlrouter.constants import AIOHTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.zmq_discovery import ZMQServiceDiscovery


logger = get_logger('dlrouter.backends.vllm')

# Default connection pool limits
DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100


class VLLMPDConfig(BaseModel):
    """vLLM PD disaggregation config.

    This config is used when serving_strategy is DISTSERVE
    and backend is vLLM. vLLM PD mode uses ZMQ for service
    discovery and request_id encoding for KV cache transfer.
    """

    zmq_host: str = '0.0.0.0'
    zmq_port: int = 30001
    ping_timeout_seconds: int = 5
    models: list[str] = Field(default_factory=list)


class VLLMBackend(BaseBackend):
    """Backend adapter for vLLM inference engine.

    Handles standard OpenAI-compatible API forwarding.
    Also supports vLLM PD disaggregation mode via request_id encoding.

    Uses a persistent aiohttp.ClientSession with connection pooling
    for better performance under high concurrency.
    """

    def __init__(
        self,
        pool_connections: int = DEFAULT_POOL_CONNECTIONS,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    ) -> None:
        self._timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        self._health_timeout = aiohttp.ClientTimeout(
            total=HEALTH_CHECK_TIMEOUT,
        )
        # Connection pool settings
        self._connector_kwargs = {
            'limit': pool_connections,
            'limit_per_host': pool_maxsize,
            'ttl_dns_cache': 300,  # DNS cache TTL in seconds
            'enable_cleanup_closed': True,
        }
        # Lazy-initialized session (bound to specific event loop)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None

    # -- Session management --

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a persistent aiohttp session.

        Uses double-checked locking for thread-safe lazy initialization.
        The lock is lazily created to ensure it's bound to the
        correct event loop.
        """
        # Lazily create the lock to ensure it's bound to
        # the current event loop
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
        """Close the persistent session.

        Should be called during application shutdown.
        """
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- Core forwarding --

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
            models = [m['id'] for m in data.get('data', [])]
            return models
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    async def check_health(self, node_url: str) -> bool:
        """Check vLLM node health via async request.

        Uses a temporary session to avoid event loop binding issues
        when called from different threads (e.g., health check thread).
        """
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

    # -- vLLM PD Disaggregation support --

    async def forward_with_request_id(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ) -> Any:
        """Forward request to vLLM node with X-Request-Id header.

        Used in vLLM PD mode where the request_id encodes
        the prefill and decode ZMQ addresses.

        Args:
            node_url: Target vLLM node URL.
            endpoint: API endpoint path.
            request_data: Request payload.
            request_id: Encoded request ID with PD addresses.

        Returns:
            Response text.
        """
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
        """Stream-forward request to vLLM node with X-Request-Id header.

        Args:
            node_url: Target vLLM node URL.
            endpoint: API endpoint path.
            request_data: Request payload.
            request_id: Encoded request ID with PD addresses.

        Yields:
            Response chunks.
        """
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

    # -- CLI argument registration --

    @classmethod
    def get_cli_args(cls) -> list[CLIArg]:
        """Return vLLM-specific CLI arguments."""
        return [
            CLIArg(
                name='zmq_host',
                type=str,
                default='0.0.0.0',
                help='ZMQ service discovery bind host (vLLM PD)',
            ),
            CLIArg(
                name='zmq_port',
                type=int,
                default=30001,
                help='ZMQ service discovery port (vLLM PD)',
            ),
            CLIArg(
                name='zmq_ping_timeout',
                type=int,
                default=5,
                help='ZMQ ping timeout in seconds (vLLM PD)',
            ),
            CLIArg(
                name='models',
                type=str,
                default=None,
                help='Comma-separated model names for vLLM PD mode',
            ),
        ]

    @classmethod
    def parse_config(cls, **kwargs) -> 'VLLMPDConfig':
        """Parse vLLM-specific config from CLI args."""
        models = []
        if kwargs.get('models'):
            models = [m.strip() for m in kwargs['models'].split(',')]
        return VLLMPDConfig(
            zmq_host=kwargs.get('zmq_host', '0.0.0.0'),
            zmq_port=kwargs.get('zmq_port', 30001),
            ping_timeout_seconds=kwargs.get('zmq_ping_timeout', 5),
            models=models,
        )

    def create_service_discovery(
        self,
        backend_config: dict[str, Any],
        node_manager: Any,
    ) -> Any:
        """Create ZMQ service discovery for vLLM PD mode.

        Args:
            backend_config: Backend-specific configuration dict.
            node_manager: The NodeManager instance.

        Returns:
            ZMQServiceDiscovery instance.
        """
        from dlrouter.core.zmq_discovery import ZMQServiceDiscovery

        config = self.parse_config(**backend_config)
        return ZMQServiceDiscovery(
            host=config.zmq_host,
            port=config.zmq_port,
            ping_timeout_seconds=config.ping_timeout_seconds,
            node_manager=node_manager,
            models=config.models,
        )

    # -- vLLM PD request handling --

    async def handle_pd_request(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool,
        service_discovery: 'ZMQServiceDiscovery',
    ) -> Any:
        """Handle request in vLLM PD disaggregation mode.

        Flow:
        1. Select P/D pair from ZMQ service discovery
        2. Build encoded request_id with ZMQ addresses
        3. Send prefill request (max_tokens=1) to P node
        4. Send decode request to D node with encoded request_id

        Args:
            request_data: The request payload.
            model_name: Requested model name.
            endpoint: API endpoint path.
            stream: Whether to stream response.
            service_discovery: ZMQ service discovery instance.

        Returns:
            StreamingResponse or JSONResponse.
        """
        # Select P/D pair
        pd_pair = service_discovery.select_pd_pair()
        if pd_pair is None:
            logger.warning('No P/D instances available')
            return JSONResponse(
                {'error': 'No prefill or decode instances available'},
                status_code=503,
            )

        (prefill_http, prefill_zmq), (decode_http, decode_zmq) = pd_pair

        # Build encoded request_id
        base_id = uuid.uuid4().hex
        request_id = service_discovery.build_request_id(prefill_zmq, decode_zmq, base_id)

        logger.info(f'vLLM PD: [HTTP:{prefill_http}, ZMQ:{prefill_zmq}] → [HTTP:{decode_http}, ZMQ:{decode_zmq}]')

        # Ensure URLs have http:// prefix
        p_url = prefill_http if prefill_http.startswith('http') else f'http://{prefill_http}'
        d_url = decode_http if decode_http.startswith('http') else f'http://{decode_http}'

        # Prefill phase (max_tokens=1)
        prefill_request = request_data.copy()
        prefill_request['max_tokens'] = 1
        if 'max_completion_tokens' in prefill_request:
            prefill_request['max_completion_tokens'] = 1

        try:
            # Send prefill request and consume response
            async for _ in self.stream_forward_with_request_id(p_url, endpoint, prefill_request, request_id):
                pass  # Consume prefill response
        except Exception as e:
            logger.error(f'vLLM PD prefill error: {e}')
            return JSONResponse(
                {'error': f'Prefill phase failed: {e}'},
                status_code=502,
            )

        # Decode phase
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
