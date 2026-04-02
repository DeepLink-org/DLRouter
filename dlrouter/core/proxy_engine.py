"""Proxy engine - orchestrates request forwarding.

Handles Hybrid (standard proxy), DistServe (LMDeploy PD disaggregation),
and vLLM PD (vLLM Prefill-Decode disaggregation) flows.
"""

import json
import uuid
from typing import TYPE_CHECKING, Any, Optional, Union

from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from dlrouter.constants import (
    ERROR_MESSAGES,
    EngineRole,
    ErrorCode,
    RoutingStrategy,
    ServingStrategy,
)
from dlrouter.core.node_manager import NodeManager
from dlrouter.logger import get_logger
from dlrouter.utils.request_key import extract_request_key


if TYPE_CHECKING:
    from dlrouter.core.zmq_discovery import ZMQServiceDiscovery
    from dlrouter.models.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )


logger = get_logger('dlrouter.proxy_engine')


class ProxyEngine:
    """Orchestrates request forwarding to backends.

    Delegates to NodeManager for routing and to the
    backend adapter for actual request forwarding.
    """

    def __init__(
        self,
        node_manager: NodeManager,
        zmq_discovery: Optional['ZMQServiceDiscovery'] = None,
    ) -> None:
        self.manager = node_manager
        self._zmq_discovery = zmq_discovery

    @property
    def backend(self):
        """Shortcut to the backend adapter."""
        return self.manager.backend

    # -- Error helpers --

    def _error_json(self, code: ErrorCode) -> dict[str, Any]:
        return {
            'error_code': code.value,
            'text': ERROR_MESSAGES[code],
        }

    def _model_not_found_response(self, model_name: str) -> bytes:
        logger.warning(f'Model not found: {model_name}')
        data = self._error_json(ErrorCode.MODEL_NOT_FOUND)
        return json.dumps(data).encode() + b'\n'

    def _timeout_response(self, node_url: str) -> bytes:
        logger.warning(f'API timeout: {node_url}')
        data = self._error_json(ErrorCode.API_TIMEOUT)
        return json.dumps(data).encode() + b'\n'

    # -- Stream wrapper --

    async def _stream_generate(
        self,
        request_data: dict[str, Any],
        node_url: str,
        endpoint: str,
    ):
        """Async generator wrapping backend stream."""
        try:
            gen = self.backend.stream_forward(node_url, endpoint, request_data)
            async for chunk in gen:
                yield chunk
        except Exception as e:
            logger.error(f'Stream error: {e}')
            yield self._timeout_response(node_url)

    # -- Hybrid mode --

    async def handle_hybrid(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool = False,
        request_key: Optional[str] = None,
    ):
        """Handle request in Hybrid mode.

        Args:
            request_data: The request payload.
            model_name: Requested model.
            endpoint: API endpoint path.
            stream: Whether to stream.
            request_key: Key for hash-based routing.

        Returns:
            StreamingResponse or JSONResponse.
        """
        node_url = self.manager.get_node_url(
            model_name,
            role=EngineRole.HYBRID,
            request_key=request_key,
        )
        if not node_url:
            return self._model_not_found_response(model_name)

        logger.info(f'Dispatching to {node_url} (model={model_name})')
        start = self.manager.pre_call(node_url)

        if stream:
            gen = self._stream_generate(request_data, node_url, endpoint)
            bg = BackgroundTasks()
            bg.add_task(self.manager.post_call, node_url, start)
            return StreamingResponse(
                gen,
                background=bg,
                media_type='text/event-stream',
            )
        try:
            text = await self.backend.forward_request(node_url, endpoint, request_data)
            self.manager.post_call(node_url, start)
            return JSONResponse(json.loads(text))
        except Exception as e:
            logger.error(f'Forward error: {e}')
            self.manager.post_call(node_url, start)
            return JSONResponse(
                self._error_json(ErrorCode.BACKEND_ERROR),
                status_code=502,
            )

    # -- DistServe (PD disaggregation) mode --

    async def handle_distserve(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool = False,
        request_key: Optional[str] = None,
    ):
        """Handle request in DistServe PD mode.

        Dispatches to the appropriate PD handler based on backend type:
        - vLLM backend with ZMQ discovery: uses handle_vllm_pd
        - LMDeploy backend: uses handle_lmdeploy_pd

        Returns:
            StreamingResponse or JSONResponse.
        """
        # Check if using vLLM backend with ZMQ service discovery
        from dlrouter.backends.vllm_backend import VLLMBackend

        if isinstance(self.backend, VLLMBackend) and self._zmq_discovery is not None:
            return await self.handle_vllm_pd(
                request_data,
                model_name,
                endpoint,
                stream,
            )

        # Otherwise use LMDeploy PD logic
        return await self.handle_lmdeploy_pd(
            request_data,
            model_name,
            endpoint,
            stream,
            request_key,
        )

    async def handle_lmdeploy_pd(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool = False,
        request_key: Optional[str] = None,
    ):
        """Handle request in LMDeploy PD mode.

        1. Send prefill request to P node
        2. Establish PD connection if needed
        3. Send decode request to D node with
           migration info

        Returns:
            StreamingResponse or JSONResponse.
        """
        if not self.backend.supports_pd_disagg():
            return JSONResponse(
                {'error': ('Current backend does not support PD disaggregation')},
                status_code=400,
            )

        pd_cfg = getattr(self.backend, 'pd_config', None)
        dummy_prefill = pd_cfg.dummy_prefill if pd_cfg else False

        # Prefill phase
        prefill_info = {}
        p_url = 'dummy:dummy'
        if not dummy_prefill:
            p_url = self.manager.get_node_url(
                model_name,
                EngineRole.PREFILL,
                request_key,
            )
            if not p_url:
                return self._model_not_found_response(model_name)
            logger.info(f'Prefill dispatched to {p_url}')
            start_p = self.manager.pre_call(p_url)
            prefill_info = (await self.backend.prefill_request(p_url, endpoint, request_data)) or {}
            self.manager.post_call(p_url, start_p)

        # Decode phase
        d_url = self.manager.get_node_url(model_name, EngineRole.DECODE, request_key)
        if not d_url:
            return self._model_not_found_response(model_name)
        logger.info(f'Decode dispatched to {d_url}')

        # PD connection
        if not dummy_prefill and not self.backend.is_connected_pd(p_url, d_url):
            await self.backend.connect_pd(p_url, d_url)

        # Add prefill url for migration
        request_data['_prefill_url'] = p_url

        start_d = self.manager.pre_call(d_url)
        if not dummy_prefill and prefill_info.get('id'):
            self.backend.shelf_prefill_session(p_url, d_url, prefill_info['id'])

        try:
            result = await self.backend.decode_request(
                d_url,
                endpoint,
                request_data,
                prefill_info,
                stream=stream,
            )
        except Exception as e:
            logger.error(f'Decode error: {e}')
            self.manager.post_call(d_url, start_d)
            return JSONResponse(
                self._error_json(ErrorCode.BACKEND_ERROR),
                status_code=502,
            )

        if stream:
            bg = BackgroundTasks()
            bg.add_task(self.manager.post_call, d_url, start_d)
            resp = StreamingResponse(
                result,
                background=bg,
                media_type='text/event-stream',
            )
        else:
            self.manager.post_call(d_url, start_d)
            resp = JSONResponse(json.loads(result))

        if not dummy_prefill and prefill_info.get('id'):
            self.backend.unshelf_prefill_session(p_url, d_url, prefill_info['id'])

        return resp

    # -- vLLM PD disaggregation mode --

    async def _stream_generate_with_request_id(
        self,
        request_data: dict[str, Any],
        node_url: str,
        endpoint: str,
        request_id: str,
    ):
        """Async generator wrapping vLLM stream with request_id."""
        try:
            gen = self.backend.stream_forward_with_request_id(
                node_url, endpoint, request_data, request_id
            )
            async for chunk in gen:
                yield chunk
        except Exception as e:
            logger.error(f'vLLM PD stream error: {e}')
            yield self._timeout_response(node_url)

    async def handle_vllm_pd(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool = False,
    ):
        """Handle request in vLLM PD disaggregation mode.

        Flow:
        1. Select P/D pair from ZMQ service discovery
        2. Build encoded request_id with ZMQ addresses
        3. Send prefill request (max_tokens=1) to P node
        4. Send decode request to D node with encoded request_id

        Returns:
            StreamingResponse or JSONResponse.
        """
        if self._zmq_discovery is None:
            return JSONResponse(
                {'error': 'ZMQ service discovery not configured for vLLM PD mode'},
                status_code=500,
            )

        # Select P/D pair
        pd_pair = self._zmq_discovery.select_pd_pair()
        if pd_pair is None:
            logger.warning('No P/D instances available')
            return JSONResponse(
                {'error': 'No prefill or decode instances available'},
                status_code=503,
            )

        (prefill_http, prefill_zmq), (decode_http, decode_zmq) = pd_pair

        # Build encoded request_id
        base_id = uuid.uuid4().hex
        request_id = self._zmq_discovery.build_request_id(
            prefill_zmq, decode_zmq, base_id
        )

        logger.info(
            f'vLLM PD: [HTTP:{prefill_http}, ZMQ:{prefill_zmq}] '
            f'→ [HTTP:{decode_http}, ZMQ:{decode_zmq}]'
        )

        # Ensure URLs have http:// prefix
        p_url = prefill_http if prefill_http.startswith('http') else f'http://{prefill_http}'
        d_url = decode_http if decode_http.startswith('http') else f'http://{decode_http}'

        # Prefill phase (max_tokens=1)
        prefill_request = request_data.copy()
        prefill_request['max_tokens'] = 1
        if 'max_completion_tokens' in prefill_request:
            prefill_request['max_completion_tokens'] = 1

        try:
            # Send prefill request
            async for _ in self._vllm_pd_forward(
                p_url, endpoint, prefill_request, request_id
            ):
                pass  # Consume prefill response
        except Exception as e:
            logger.error(f'vLLM PD prefill error: {e}')
            return JSONResponse(
                self._error_json(ErrorCode.BACKEND_ERROR),
                status_code=502,
            )

        # Decode phase
        if stream:
            gen = self._stream_generate_with_request_id(
                request_data, d_url, endpoint, request_id
            )
            return StreamingResponse(
                gen,
                media_type='text/event-stream',
            )

        try:
            text = await self.backend.forward_with_request_id(
                d_url, endpoint, request_data, request_id
            )
            return JSONResponse(json.loads(text))
        except Exception as e:
            logger.error(f'vLLM PD decode error: {e}')
            return JSONResponse(
                self._error_json(ErrorCode.BACKEND_ERROR),
                status_code=502,
            )

    async def _vllm_pd_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ):
        """Forward request to vLLM node with request_id (async generator)."""
        gen = self.backend.stream_forward_with_request_id(
            node_url, endpoint, request_data, request_id
        )
        async for chunk in gen:
            yield chunk

    # -- Unified dispatch --

    def _extract_request_key_if_needed(
        self,
        raw_request: Optional[Request],
        body: Union['ChatCompletionRequest', 'CompletionRequest', None],
    ) -> Optional[str]:
        """Extract request key only for consistent hash strategy.

        Args:
            raw_request: The raw HTTP request (for headers).
            body: The parsed request body.

        Returns:
            Request key if using consistent hash, None otherwise.
        """
        if self.manager.routing_strategy != RoutingStrategy.CONSISTENT_HASH:
            return None
        return extract_request_key(raw_request, body)

    async def dispatch(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool = False,
        raw_request: Optional[Request] = None,
        body: Union['ChatCompletionRequest', 'CompletionRequest', None] = None,
    ):
        """Dispatch request based on serving strategy.

        Args:
            request_data: The request payload dict.
            model_name: Requested model name.
            endpoint: API endpoint path.
            stream: Whether to stream response.
            raw_request: Raw HTTP request for header extraction.
            body: Parsed request body for field extraction.

        Returns:
            StreamingResponse or JSONResponse.
        """
        # Extract request key only for consistent hash
        request_key = self._extract_request_key_if_needed(raw_request, body)
        strategy = self.manager.serving_strategy
        if strategy == ServingStrategy.HYBRID:
            return await self.handle_hybrid(
                request_data,
                model_name,
                endpoint,
                stream,
                request_key,
            )
        if strategy == ServingStrategy.DISTSERVE:
            return await self.handle_distserve(
                request_data,
                model_name,
                endpoint,
                stream,
                request_key,
            )
        raise ValueError(f'Unknown serving strategy: {strategy}')
