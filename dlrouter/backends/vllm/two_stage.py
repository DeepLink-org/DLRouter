"""Two-stage PD executor for the vLLM backend."""

import json
from typing import Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.vllm.kv_transfer import KVTransferAdapter
from dlrouter.backends.vllm.pair_selection import VLLMPairSelector
from dlrouter.backends.vllm.request_state import VLLMTwoStageRequestState
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.two_stage')


class VLLMTwoStagePDExecutor:
    """Execute a prefill/decode two-stage flow for vLLM."""

    def __init__(
        self,
        backend: Any,
        adapter: KVTransferAdapter,
        pair_selector: VLLMPairSelector | None = None,
    ) -> None:
        self.backend = backend
        self.adapter = adapter
        self.pair_selector = pair_selector or VLLMPairSelector()

    async def execute(
        self,
        request_data: dict[str, Any],
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        node_manager = context.node_manager

        pd_pair = self.pair_selector.select_pair(
            node_manager=node_manager,
            model_name=request_data.get('model', ''),
            request_key=context.request_key,
        )
        if pd_pair is None:
            return JSONResponse(
                {'error': 'No prefill or decode instances available'},
                status_code=503,
            )

        prefill_url, decode_url = pd_pair
        request_id = self.adapter.build_request_id(prefill_url, decode_url, node_manager)
        state = VLLMTwoStageRequestState(
            request_id=request_id,
            prefill_url=prefill_url,
            decode_url=decode_url,
        )

        # -- Prefill phase --
        prefill_start = self._pre_call(node_manager, state.prefill_url)

        prefill_request = self.adapter.build_prefill_request(
            request_data,
            request_id,
            state.aborted_request_ids,
        )
        try:
            prefill_text = await self.backend.forward_with_request_id(
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

        # -- Decode phase --
        if stream:
            return StreamingResponse(
                self._stream_decode(state, endpoint, decode_request, node_manager),
                media_type='text/event-stream',
            )

        decode_start = self._pre_call(node_manager, state.decode_url)
        try:
            text = await self.backend.forward_with_request_id(
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
        state: VLLMTwoStageRequestState,
        endpoint: str,
        decode_request: dict[str, Any],
        node_manager: Optional[Any] = None,
    ):
        decode_start = self._pre_call(node_manager, state.decode_url)
        try:
            async for chunk in self.backend.stream_forward_with_request_id(
                state.decode_url,
                endpoint,
                decode_request,
                state.request_id,
            ):
                if chunk and not state.prefill_kv_released:
                    state.prefill_kv_released = True

                yield chunk
        except Exception:
            state.mark_aborted()
            self._post_call(node_manager, state.decode_url, decode_start)
            raise

        self._post_call(node_manager, state.decode_url, decode_start)

    # -- Lifecycle helpers --

    @staticmethod
    def _pre_call(node_manager: Optional[Any], node_url: str) -> Optional[float]:
        """Track request start on a node (for load-aware routing)."""
        if node_manager is None:
            return None
        try:
            return node_manager.pre_call(node_url)
        except (KeyError, AttributeError):
            logger.debug(f'pre_call skipped: {node_url} not in NodeManager')
            return None

    @staticmethod
    def _post_call(
        node_manager: Optional[Any],
        node_url: str,
        start: Optional[float],
    ) -> None:
        """Track request end on a node (for load-aware routing)."""
        if node_manager is None or start is None:
            return
        try:
            node_manager.post_call(node_url, start)
        except (KeyError, AttributeError):
            logger.debug(f'post_call skipped: {node_url} not in NodeManager')
