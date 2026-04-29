"""Shared dual-dispatch PD executor."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.pd.selection import PDPairSelector, no_pd_pair_response
from dlrouter.core.node_lifecycle import post_call, pre_call


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dlrouter.backends.base import PDRequestContext
    from dlrouter.backends.pd.protocols import DualDispatchAdapter, DualDispatchTransport


class DualDispatchExecutor:
    """Execute PD by sending the same routed request to prefill and decode."""

    def __init__(
        self,
        transport: DualDispatchTransport,
        adapter: DualDispatchAdapter,
        pair_selector: PDPairSelector | None = None,
    ) -> None:
        self.transport = transport
        self.adapter = adapter
        self.pair_selector = pair_selector or PDPairSelector()

    async def execute(
        self,
        request_data: dict[str, Any],
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Execute a dual-dispatch PD request."""
        pd_pair = self.pair_selector.select_pair(
            node_manager=context.node_manager,
            model_name=request_data.get('model', ''),
            request_key=context.request_key,
        )
        if pd_pair is None:
            return no_pd_pair_response()

        prefill_url = pd_pair.prefill_url
        decode_url = pd_pair.decode_url
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
        except (asyncio.CancelledError, Exception):
            prefill_task.cancel()
            with suppress(asyncio.CancelledError):
                await prefill_task
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
            return await self.transport.forward_request(node_url, endpoint, request_data)
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
            async for chunk in self.transport.stream_forward(
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
        return pre_call(node_manager, node_url)

    @staticmethod
    def _post_call(node_manager: Any, node_url: str, start: float | None) -> None:
        """Track request end on a node for load-aware routing."""
        post_call(node_manager, node_url, start)
