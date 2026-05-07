"""Shared two-stage transfer PD executor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.pd.selection import PDPairSelector, no_pd_pair_response
from dlrouter.backends.pd.state import TwoStageRequestState
from dlrouter.core.node_lifecycle import post_call, pre_call


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dlrouter.backends.base import PDRequestContext
    from dlrouter.backends.pd.protocols import TwoStageTransferAdapter, TwoStageTransferTransport
    from dlrouter.core.node_manager import NodeManager


class TwoStageTransferExecutor:
    """Execute a prefill/decode two-stage transfer flow."""

    def __init__(
        self,
        transport: TwoStageTransferTransport,
        adapter: TwoStageTransferAdapter,
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
        """Execute a two-stage transfer PD request."""
        node_manager = context.node_manager

        pd_pair = self.pair_selector.select_pair(
            node_manager=node_manager,
            model_name=request_data.get('model', ''),
            request_key=context.request_key,
        )
        if pd_pair is None:
            return no_pd_pair_response()

        prefill_url = pd_pair.prefill_url
        decode_url = pd_pair.decode_url
        request_id = self.adapter.build_request_id(prefill_url, decode_url, node_manager)
        state = TwoStageRequestState(
            request_id=request_id,
            prefill_url=prefill_url,
            decode_url=decode_url,
        )

        prefill_start = self._pre_call(node_manager, state.prefill_url)

        prefill_request = self.adapter.build_prefill_request(
            request_data,
            request_id,
            state.aborted_request_ids,
        )
        try:
            prefill_text = await self.transport.forward_with_request_id(
                state.prefill_url,
                endpoint,
                prefill_request,
                request_id,
            )
        except Exception:
            self._post_call(node_manager, state.prefill_url, prefill_start)
            raise

        self._post_call(node_manager, state.prefill_url, prefill_start)

        prefill_json = json.loads(prefill_text)
        transfer_context = self.adapter.extract_transfer_context(prefill_json)
        if transfer_context is not None:
            decode_request = self.adapter.inject_decode_request(
                request_data,
                transfer_context,
            )
        else:
            decode_request = request_data.copy()

        if stream:
            return StreamingResponse(
                self._stream_decode(state, endpoint, decode_request, node_manager),
                media_type='text/event-stream',
            )

        decode_start = self._pre_call(node_manager, state.decode_url)
        try:
            text = await self.transport.forward_with_request_id(
                state.decode_url,
                endpoint,
                decode_request,
                request_id,
            )
        except Exception:
            self._post_call(node_manager, state.decode_url, decode_start)
            raise

        self._post_call(node_manager, state.decode_url, decode_start)
        return JSONResponse(json.loads(text))

    async def _stream_decode(
        self,
        state: TwoStageRequestState,
        endpoint: str,
        decode_request: dict[str, Any],
        node_manager: NodeManager,
    ) -> AsyncIterator[bytes]:
        decode_start = self._pre_call(node_manager, state.decode_url)
        try:
            async for chunk in self.transport.stream_forward_with_request_id(
                state.decode_url,
                endpoint,
                decode_request,
                state.request_id,
            ):
                yield chunk
        except BaseException:
            state.mark_aborted()
            self._post_call(node_manager, state.decode_url, decode_start)
            raise

        self._post_call(node_manager, state.decode_url, decode_start)

    @staticmethod
    def _pre_call(node_manager: NodeManager, node_url: str) -> float | None:
        """Track request start on a node for load-aware routing."""
        return pre_call(node_manager, node_url)

    @staticmethod
    def _post_call(
        node_manager: NodeManager,
        node_url: str,
        start: float | None,
    ) -> None:
        """Track request end on a node for load-aware routing."""
        post_call(node_manager, node_url, start)
