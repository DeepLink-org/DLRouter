"""LMDeploy backend adapter.

Supports both Hybrid and DistServe (PD disaggregation)
serving strategies.
"""

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from typing import Any, Optional

import aiohttp
import requests

from dlrouter.backends.base import BaseBackend
from dlrouter.config import LMDeployPDConfig
from dlrouter.constants import AIOHTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.backends.lmdeploy')

# Default connection pool limits
DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100


class LMDeployBackend(BaseBackend):
    """Backend adapter for LMDeploy inference engine.

    Handles both standard (Hybrid) forwarding and
    PD disaggregation (DistServe) flows.

    Uses a persistent aiohttp.ClientSession with connection pooling
    for better performance under high concurrency.
    """

    def __init__(
        self,
        pd_config: Optional[LMDeployPDConfig] = None,
        pool_connections: int = DEFAULT_POOL_CONNECTIONS,
        pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    ) -> None:
        self.pd_config = pd_config or LMDeployPDConfig()
        self._timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        self._health_timeout = aiohttp.ClientTimeout(
            total=HEALTH_CHECK_TIMEOUT,
        )
        # Connection pool settings
        self._connector_kwargs = {
            'limit': pool_connections,
            'limit_per_host': pool_maxsize,
            'ttl_dns_cache': 300,
            'enable_cleanup_closed': True,
        }
        # Lazy-initialized session (bound to specific event loop)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None
        # PD connection pool (lazy import)
        self._pd_pool = None

    def _get_pd_pool(self):
        """Lazy-init PD connection pool."""
        if self._pd_pool is None:
            try:
                from lmdeploy.pytorch.disagg.conn.proxy_conn import (
                    PDConnectionPool,
                )

                self._pd_pool = PDConnectionPool()
            except ImportError:
                logger.warning(
                    'lmdeploy PD disagg not available. Install lmdeploy for PD support.',
                )
                self._pd_pool = None
        return self._pd_pool

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

    def deregister_node(self, node_url: str) -> None:
        """Remove node from PD connection pool."""
        pool = self._get_pd_pool()
        if pool is not None:
            pool.dereg_instance(node_url)

    # -- Core forwarding --

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward request to LMDeploy node."""
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
        """Stream-forward request to LMDeploy node."""
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
        """Fetch available models from LMDeploy node."""
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
        """Check LMDeploy node health via async request.

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

    # -- PD Disaggregation support --

    def supports_pd_disagg(self) -> bool:
        """LMDeploy supports PD disaggregation."""
        return True

    async def prefill_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Send prefill-only request to P node."""
        prefill_data = copy.deepcopy(request_data)
        prefill_data['max_tokens'] = 1
        prefill_data['max_completion_tokens'] = 1
        prefill_data['stream'] = False
        prefill_data['with_cache'] = True
        prefill_data['preserve_cache'] = True

        try:
            text = await self.forward_request(node_url, endpoint, prefill_data)
            return json.loads(text)
        except Exception as e:
            logger.error(f'Prefill request failed on {node_url}: {e}')
            return None

    async def decode_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        prefill_info: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Send decode request with migration info."""
        try:
            from lmdeploy.pytorch.disagg.conn.protocol import (
                MigrationProtocol,
                MigrationRequest,
            )
        except ImportError:
            logger.error('lmdeploy disagg not available.')
            raise

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_ids = prefill_info.get('remote_token_ids')
        remote_token_id = remote_token_ids[-1] if remote_token_ids else 0
        dummy = self.pd_config.dummy_prefill
        protocol = MigrationProtocol[self.pd_config.migration_protocol]

        migration = MigrationRequest(
            protocol=protocol,
            remote_engine_id=request_data.get('_prefill_url', ''),
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=dummy,
        )
        request_data['migration_request'] = migration.model_dump(mode='json')

        if stream:
            return self.stream_forward(node_url, endpoint, request_data)
        return await self.forward_request(node_url, endpoint, request_data)

    def is_connected_pd(self, p_url: str, d_url: str) -> bool:
        """Check if PD connection exists between nodes."""
        pool = self._get_pd_pool()
        return pool is not None and pool.is_connected(p_url, d_url)

    async def connect_pd(self, p_url: str, d_url: str) -> None:
        """Establish PD connection between nodes."""
        pool = self._get_pd_pool()
        if pool is None:
            logger.warning('No PD pool available.')
            return
        if pool.is_connected(p_url, d_url):
            return
        try:
            from lmdeploy.pytorch.disagg.config import (
                DistServeRDMAConfig,
                RDMALinkType,
            )
            from lmdeploy.pytorch.disagg.conn.protocol import (
                MigrationProtocol,
            )
            from lmdeploy.pytorch.disagg.messages import (
                PDConnectionMessage,
            )

            rdma_cfg = DistServeRDMAConfig(
                with_gdr=self.pd_config.with_gdr,
                link_type=RDMALinkType[self.pd_config.link_type],
            )
            protocol = MigrationProtocol[self.pd_config.migration_protocol]
            msg = PDConnectionMessage(
                p_url=p_url,
                d_url=d_url,
                protocol=protocol,
                rdma_config=rdma_cfg,
            )
            await pool.connect(msg)
        except Exception as e:
            logger.error(f'PD connection error ({p_url} -> {d_url}): {e}')

    def shelf_prefill_session(
        self,
        p_url: str,
        d_url: str,
        session_id: str,
    ) -> None:
        """Shelf prefill session in PD pool."""
        pool = self._get_pd_pool()
        if pool:
            pool.shelf_prefill_session((p_url, d_url), session_id)

    def unshelf_prefill_session(
        self,
        p_url: str,
        d_url: str,
        session_id: str,
    ) -> None:
        """Unshelf prefill session in PD pool."""
        pool = self._get_pd_pool()
        if pool:
            pool.unshelf_prefill_session((p_url, d_url), session_id)
