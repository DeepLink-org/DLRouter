"""vLLM backend adapter.

Supports standard OpenAI-compatible API forwarding
for vLLM inference engine.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Optional

import aiohttp
import requests

from dlrouter.backends.base import BaseBackend
from dlrouter.constants import AIOHTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.backends.vllm')

# Default connection pool limits
DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100


class VLLMBackend(BaseBackend):
    """Backend adapter for vLLM inference engine.

    Handles standard OpenAI-compatible API forwarding.
    vLLM does not support PD disaggregation.

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
