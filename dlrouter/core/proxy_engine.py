"""Proxy engine - orchestrates request forwarding.

Handles Hybrid (standard proxy) and DistServe (PD disaggregation) flows.
Backend-specific PD logic is delegated to the backend's handle_pd_request().
"""

import json
from typing import Any, Optional, Union

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
from dlrouter.models.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from dlrouter.utils.request_key import extract_request_key


logger = get_logger('dlrouter.proxy_engine')


class ProxyEngine:
    """Orchestrates request forwarding to backends.

    Delegates to NodeManager for routing and to the
    backend adapter for actual request forwarding.
    """

    def __init__(
        self,
        node_manager: NodeManager,
        service_discovery: Optional[Any] = None,
    ) -> None:
        self.manager = node_manager
        self._service_discovery = service_discovery

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

        Delegates to backend.handle_pd_request() if service discovery is available,
        otherwise falls back to LMDeploy PD logic.

        Returns:
            StreamingResponse or JSONResponse.
        """
        # If backend has service discovery, delegate to backend's PD handler
        if self._service_discovery is not None:
            return await self.backend.handle_pd_request(
                request_data,
                model_name,
                endpoint,
                stream,
                self._service_discovery,
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

    # -- Unified dispatch --

    def _extract_request_key_if_needed(
        self,
        raw_request: Optional[Request],
        body: Union[ChatCompletionRequest, CompletionRequest, None],
    ) -> Optional[str]:
        """Extract request key for routing strategies that need it.

        For CONSISTENT_HASH: extracts session/user keys from headers/body.
        For PREFIX_CACHE: extracts the prompt text for prefix matching.

        Args:
            raw_request: The raw HTTP request (for headers).
            body: The parsed request body.

        Returns:
            Request key/prompt if using a strategy that requires it, None otherwise.
        """
        strategy = self.manager.routing_strategy

        if strategy == RoutingStrategy.CONSISTENT_HASH:
            return extract_request_key(raw_request, body)

        if strategy == RoutingStrategy.PREFIX_CACHE:
            return self._extract_prompt_for_prefix_cache(body)

        return None

    def _extract_prompt_for_prefix_cache(
        self,
        body: Union['ChatCompletionRequest', 'CompletionRequest', None],
    ) -> Optional[str]:
        """Extract prompt text from request body for prefix cache matching.

        Handles both ChatCompletionRequest (messages) and CompletionRequest (prompt).

        Args:
            body: The parsed request body.

        Returns:
            The prompt string for prefix matching, or None if not found.
        """
        if body is None:
            return None

        # Handle ChatCompletionRequest - extract from messages
        if hasattr(body, 'messages') and body.messages is not None:
            messages = body.messages
            if isinstance(messages, str):
                return messages
            if isinstance(messages, list) and len(messages) > 0:
                # Concatenate all message content to form the prompt
                parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get('content', '')
                        if content:
                            parts.append(str(content))
                    elif hasattr(msg, 'content'):
                        if msg.content:
                            parts.append(str(msg.content))
                return '\n'.join(parts) if parts else None

        # Handle CompletionRequest - extract from prompt field
        if hasattr(body, 'prompt') and body.prompt is not None:
            prompt = body.prompt
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list) and len(prompt) > 0:
                # Join list of prompts
                return '\n'.join(str(p) for p in prompt)

        return None

    async def dispatch(
        self,
        request_data: dict[str, Any],
        model_name: str,
        endpoint: str,
        stream: bool = False,
        raw_request: Optional[Request] = None,
        body: Union[ChatCompletionRequest, CompletionRequest, None] = None,
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
