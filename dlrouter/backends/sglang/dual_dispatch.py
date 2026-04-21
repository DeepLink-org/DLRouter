"""Dual-dispatch PD executor for the SGLang backend."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.sglang.pair_selection import SGLangPairSelector
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dlrouter.backends.base import PDRequestContext
    from dlrouter.backends.sglang.transfer import SGLangBootstrapAdapter


logger = get_logger('dlrouter.sglang.dual_dispatch')


class SGLangDualDispatchExecutor:
    """Execute SGLang PD by sending the routed request to P and D together."""

    def __init__(
        self,
        backend: Any,
        adapter: SGLangBootstrapAdapter,
        pair_selector: SGLangPairSelector | None = None,
    ) -> None:
        self.backend = backend
        self.adapter = adapter
        self.pair_selector = pair_selector or SGLangPairSelector()

    async def execute(
        self,
        request_data: dict[str, Any],
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Execute SGLang PD dual dispatch."""
        pd_pair = self.pair_selector.select_pair(
            node_manager=context.node_manager,
            model_name=request_data.get('model', ''),
            request_key=context.request_key,
        )
        if pd_pair is None:
            return JSONResponse(
                {'error': 'No prefill or decode instances available'},
                status_code=503,
            )

        prefill_url, decode_url = pd_pair
        routed_request = self.adapter.build_request(
            request_data,
            prefill_url=prefill_url,
            endpoint=endpoint,
        )

        if stream:
            return StreamingResponse(
                self._stream_dual_dispatch(
                    prefill_url=prefill_url,
                    decode_url=decode_url,
                    endpoint=endpoint,
                    routed_request=routed_request,
                    context=context,
                ),
                media_type='text/event-stream',
            )

        _, decode_text = await asyncio.gather(
            self._forward_tracked(
                prefill_url,
                endpoint,
                routed_request,
                context.node_manager,
            ),
            self._forward_tracked(
                decode_url,
                endpoint,
                routed_request,
                context.node_manager,
            ),
        )
        return JSONResponse(json.loads(decode_text))

    async def _stream_dual_dispatch(
        self,
        *,
        prefill_url: str,
        decode_url: str,
        endpoint: str,
        routed_request: dict[str, Any],
        context: PDRequestContext,
    ) -> AsyncIterator[bytes]:
        prefill_task = asyncio.create_task(
            self._forward_tracked(
                prefill_url,
                endpoint,
                routed_request,
                context.node_manager,
            )
        )
        try:
            async for chunk in self._stream_tracked(
                decode_url,
                endpoint,
                routed_request,
                context.node_manager,
            ):
                yield chunk
            await prefill_task
        except Exception:
            prefill_task.cancel()
            raise

    async def _forward_tracked(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        node_manager: Any,
    ) -> Any:
        start = self._pre_call(node_manager, node_url)
        try:
            return await self.backend.forward_request(node_url, endpoint, request_data)
        finally:
            self._post_call(node_manager, node_url, start)

    async def _stream_tracked(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        node_manager: Any,
    ) -> AsyncIterator[bytes]:
        start = self._pre_call(node_manager, node_url)
        try:
            async for chunk in self.backend.stream_forward(
                node_url,
                endpoint,
                request_data,
            ):
                yield chunk
        finally:
            self._post_call(node_manager, node_url, start)

    @staticmethod
    def _pre_call(node_manager: Any, node_url: str) -> float | None:
        """Track request start on a node for load-aware routing."""
        try:
            return node_manager.pre_call(node_url)
        except (KeyError, AttributeError):
            logger.debug(f'pre_call skipped: {node_url} not in NodeManager')
            return None

    @staticmethod
    def _post_call(node_manager: Any, node_url: str, start: float | None) -> None:
        """Track request end on a node for load-aware routing."""
        if start is None:
            return
        try:
            node_manager.post_call(node_url, start)
        except (KeyError, AttributeError):
            logger.debug(f'post_call skipped: {node_url} not in NodeManager')
