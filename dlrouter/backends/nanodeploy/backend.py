"""NanoDeploy backend adapter.

Forwards OpenAI-compatible HTTP to NanoDeploy ``serve`` nodes. When
``--ctrl_address`` is set, discovers nodes via dlslime-ctrl (entity kind
``nanodeploy``).
"""

import json
from typing import TYPE_CHECKING, Any, Optional

import aiohttp
import requests
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.base import BaseBackend, CLIArg, PDRequestContext
from dlrouter.backends.http import BackendHTTPTransportMixin, StreamFraming
from dlrouter.backends.nanodeploy.config import NanoDeployConfig
from dlrouter.constants import (
    AIOHTTP_TIMEOUT,
    ERROR_MESSAGES,
    HEALTH_CHECK_TIMEOUT,
    EngineRole,
    ErrorCode,
    ServiceDiscoveryMode,
)
from dlrouter.core.dp_url import normalize_dp_aware_url
from dlrouter.core.node_lifecycle import post_call, pre_call
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager
    from dlrouter.core.service_discovery.base import BaseServiceDiscovery


logger = get_logger('dlrouter.backends.nanodeploy')

DEFAULT_POOL_CONNECTIONS = 100
DEFAULT_POOL_MAXSIZE = 100

# DLRouter adds routing metadata; NanoDeploy serve only needs generation fields.
# ``kv_transfer_params`` carries the PD handoff (do_remote_decode / migration).
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
        'kv_transfer_params',
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
        """NanoDeploy supports two-stage PD disaggregation over HTTP."""
        return True

    @staticmethod
    def _error_json(code: ErrorCode) -> dict[str, Any]:
        return {'error_code': code.value, 'text': ERROR_MESSAGES[code]}

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
        """Two-stage PD: prefill (1 token + KV) -> decode (RDMA-pull + stream).

        Stage 1 asks a prefill node to run a single-token prefill and return an
        opaque ``kv_transfer_params.migration`` payload (a serialized prefilled
        sequence pointing at the prefill engine's KV blocks). Stage 2 hands that
        payload to a decode node, which RDMA-pulls the KV cache and generates the
        full completion. The prefill KV blocks are released afterwards via
        ``POST /pd/free``.
        """
        node_manager = context.node_manager
        request_key = context.request_key

        p_url = node_manager.get_node_url(model_name, EngineRole.PREFILL, request_key)
        if not p_url:
            return self._model_not_found_response(model_name)
        d_url = node_manager.get_node_url(model_name, EngineRole.DECODE, request_key)
        if not d_url:
            return self._model_not_found_response(model_name)

        logger.info(f'PD prefill={p_url} decode={d_url}')

        # ---- Stage 1: prefill ----
        start_p = pre_call(node_manager, p_url)
        try:
            prefill_info = await self._prefill_request(p_url, endpoint, request_data)
        finally:
            post_call(node_manager, p_url, start_p)

        if prefill_info is None:
            return self._backend_error_response()

        kv = prefill_info.get('kv_transfer_params') or {}
        migration = kv.get('migration')
        seq_id = kv.get('seq_id')
        if not migration:
            # No KV to migrate: the prefill node fully finished the request
            # locally (e.g. the first sampled token was EOS, so the scheduler
            # marked the sequence FINISHED instead of TO_BE_MIGRATED). Return
            # its completion directly instead of handing off to a decode node.
            if prefill_info.get('choices'):
                logger.info('Prefill produced a full completion; skipping decode')
                if stream:
                    return StreamingResponse(
                        self._completion_as_sse(prefill_info),
                        media_type='text/event-stream',
                    )
                return JSONResponse(prefill_info)
            logger.error('Prefill returned no migration payload')
            return self._backend_error_response()

        # ---- Stage 2: decode ----
        decode_data = _sanitize_chat_payload(request_data)
        decode_data['kv_transfer_params'] = {'migration': migration}
        decode_data['stream'] = stream

        free_ids = [seq_id] if seq_id is not None else []
        start_d = pre_call(node_manager, d_url)

        if stream:
            async def _stream():
                async for chunk in self.stream_forward(d_url, endpoint, decode_data):
                    yield chunk

            bg = BackgroundTasks()
            bg.add_task(post_call, node_manager, d_url, start_d)
            if free_ids:
                bg.add_task(self._free_prefill, p_url, free_ids)
            return StreamingResponse(
                _stream(),
                background=bg,
                media_type='text/event-stream',
            )

        try:
            text = await self.forward_request(d_url, endpoint, decode_data)
        except Exception as e:
            logger.error(f'Decode error on {d_url}: {e}')
            return self._backend_error_response()
        finally:
            post_call(node_manager, d_url, start_d)
            if free_ids:
                await self._free_prefill(p_url, free_ids)
        return JSONResponse(json.loads(text))

    async def _prefill_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Run prefill and return the migration payload.

        We do not clamp ``max_tokens`` to 1: a NanoDeploy ``mode="prefill"``
        engine already emits exactly one token before handing the sequence off
        for migration, and the user's ``max_tokens`` must survive into the
        migrated sequence so the decode engine resumes with the right budget.
        """
        data = _sanitize_chat_payload(request_data)
        data['stream'] = False
        data['kv_transfer_params'] = {'do_remote_decode': True}
        try:
            text = await self.forward_request(node_url, endpoint, data)
            return json.loads(text)
        except Exception as e:
            logger.error(f'Prefill request failed on {node_url}: {e}')
            return None

    @staticmethod
    async def _completion_as_sse(completion: dict[str, Any]):
        """Emit a finished (non-migrated) completion as a one-shot SSE stream.

        Used when the prefill node fully answered the request so there is no
        decode handoff, but the client asked for a streaming response.
        """
        obj = completion.get('object') or ''
        choice = (completion.get('choices') or [{}])[0]
        finish_reason = choice.get('finish_reason', 'stop')
        if obj == 'chat.completion':
            content = (choice.get('message') or {}).get('content', '')
            chunk = {
                'id': completion.get('id'),
                'object': 'chat.completion.chunk',
                'created': completion.get('created'),
                'model': completion.get('model'),
                'choices': [
                    {
                        'index': 0,
                        'delta': {'role': 'assistant', 'content': content},
                        'finish_reason': finish_reason,
                    }
                ],
            }
        else:
            chunk = {
                'id': completion.get('id'),
                'object': 'text_completion',
                'created': completion.get('created'),
                'model': completion.get('model'),
                'choices': [
                    {
                        'index': 0,
                        'text': choice.get('text', ''),
                        'finish_reason': finish_reason,
                    }
                ],
            }
        yield f'data: {json.dumps(chunk)}\n\n'.encode()
        yield b'data: [DONE]\n\n'

    async def _free_prefill(self, node_url: str, seq_ids: list[int]) -> None:
        """Release prefill-side MIGRATE KV blocks via POST /pd/free."""
        try:
            session = await self._get_session()
            url = normalize_dp_aware_url(node_url) + '/pd/free'
            async with session.post(url, json={'seq_ids': seq_ids}) as resp:
                await resp.read()
        except Exception as e:  # noqa: BLE001
            logger.warning(f'PD free failed on {node_url} for {seq_ids}: {e}')

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
