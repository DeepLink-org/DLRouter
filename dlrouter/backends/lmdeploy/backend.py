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
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.base import BaseBackend, CLIArg, PDRequestContext
from dlrouter.backends.lmdeploy.config import LMDeployPDConfig
from dlrouter.constants import (
    AIOHTTP_TIMEOUT,
    ERROR_MESSAGES,
    HEALTH_CHECK_TIMEOUT,
    EngineRole,
    ErrorCode,
    ServiceDiscoveryMode,
)
from dlrouter.core.node_lifecycle import post_call, pre_call
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.backends.lmdeploy')

# Default connection pool limits
DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100


class LMDeployBackend(BaseBackend):
    """Backend adapter for LMDeploy inference engine."""

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
        self._connector_kwargs = {
            'limit': pool_connections,
            'limit_per_host': pool_maxsize,
            'ttl_dns_cache': 300,
            'enable_cleanup_closed': True,
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: Optional[asyncio.Lock] = None
        self._pd_pool = None

    @classmethod
    def create(cls, parsed_config: Optional[LMDeployPDConfig] = None) -> 'LMDeployBackend':
        """Create an LMDeploy backend from parsed configuration."""
        return cls(pd_config=parsed_config)

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

    def deregister_node(self, node_url: str) -> None:
        """Remove node from PD connection pool."""
        pool = self._get_pd_pool()
        if pool is not None:
            pool.dereg_instance(node_url)

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
            return [m['id'] for m in data.get('data', [])]
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    async def check_health(self, node_url: str) -> bool:
        """Check LMDeploy node health via async request."""
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

    def supports_pd_disagg(self) -> bool:
        """LMDeploy supports PD disaggregation."""
        return True

    def preferred_discovery_mode(
        self,
        backend_config: dict[str, Any],
    ) -> Optional[ServiceDiscoveryMode]:
        """LMDeploy DistServe relies on external P/D node registration."""
        return None

    @staticmethod
    def _error_json(code: ErrorCode) -> dict[str, Any]:
        return {
            'error_code': code.value,
            'text': ERROR_MESSAGES[code],
        }

    def _model_not_found_response(self, model_name: str) -> JSONResponse:
        logger.warning(f'Model not found: {model_name}')
        return JSONResponse(
            self._error_json(ErrorCode.MODEL_NOT_FOUND),
            status_code=404,
        )

    def _backend_error_response(self) -> JSONResponse:
        return JSONResponse(
            self._error_json(ErrorCode.BACKEND_ERROR),
            status_code=502,
        )

    async def handle_pd_request(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Handle request in LMDeploy PD mode."""
        node_manager = context.node_manager
        request_key = context.request_key
        dummy_prefill = self.pd_config.dummy_prefill

        prefill_info: dict[str, Any] = {}
        p_url = 'dummy:dummy'
        if not dummy_prefill:
            p_url = node_manager.get_node_url(
                model_name,
                EngineRole.PREFILL,
                request_key,
            )
            if not p_url:
                return self._model_not_found_response(model_name)

            logger.info(f'Prefill dispatched to {p_url}')
            start_p = pre_call(node_manager, p_url)
            try:
                prefill_info = (await self.prefill_request(p_url, endpoint, request_data)) or {}
            finally:
                post_call(node_manager, p_url, start_p)

        d_url = node_manager.get_node_url(model_name, EngineRole.DECODE, request_key)
        if not d_url:
            return self._model_not_found_response(model_name)
        logger.info(f'Decode dispatched to {d_url}')

        if not dummy_prefill and not self.is_connected_pd(p_url, d_url):
            await self.connect_pd(p_url, d_url)

        decode_request_data = copy.deepcopy(request_data)
        decode_request_data['_prefill_url'] = p_url

        start_d = pre_call(node_manager, d_url)
        should_unshelf = bool(not dummy_prefill and prefill_info.get('id'))
        if should_unshelf:
            self.shelf_prefill_session(p_url, d_url, prefill_info['id'])

        try:
            result = await self.decode_request(
                d_url,
                endpoint,
                decode_request_data,
                prefill_info,
                stream=stream,
            )
        except Exception as e:
            logger.error(f'Decode error: {e}')
            post_call(node_manager, d_url, start_d)
            if should_unshelf:
                self.unshelf_prefill_session(p_url, d_url, prefill_info['id'])
            return self._backend_error_response()

        if stream:
            bg = BackgroundTasks()
            bg.add_task(post_call, node_manager, d_url, start_d)
            if should_unshelf:
                bg.add_task(self.unshelf_prefill_session, p_url, d_url, prefill_info['id'])
            return StreamingResponse(
                result,
                background=bg,
                media_type='text/event-stream',
            )

        post_call(node_manager, d_url, start_d)
        if should_unshelf:
            self.unshelf_prefill_session(p_url, d_url, prefill_info['id'])
        return JSONResponse(json.loads(result))

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

    @classmethod
    def get_cli_args(cls) -> list[CLIArg]:
        """Return LMDeploy-specific CLI arguments."""
        return [
            CLIArg(
                name='migration_protocol',
                type=str,
                default='RDMA',
                help='PD migration protocol (LMDeploy)',
            ),
            CLIArg(
                name='link_type',
                type=str,
                default='RoCE',
                help='RDMA link type (LMDeploy)',
                choices=['RoCE', 'IB'],
            ),
            CLIArg(
                name='with_gdr',
                type=bool,
                default=True,
                help='Enable GPU Direct RDMA (LMDeploy)',
            ),
            CLIArg(
                name='dummy_prefill',
                type=bool,
                default=False,
                help='Use dummy prefill for testing (LMDeploy)',
            ),
        ]

    @classmethod
    def parse_config(cls, **kwargs) -> LMDeployPDConfig:
        """Parse LMDeploy-specific config from CLI args."""
        return LMDeployPDConfig(
            migration_protocol=kwargs.get('migration_protocol', 'RDMA'),
            link_type=kwargs.get('link_type', 'RoCE'),
            with_gdr=kwargs.get('with_gdr', True),
            dummy_prefill=kwargs.get('dummy_prefill', False),
        )
