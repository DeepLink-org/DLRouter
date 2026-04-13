"""Two-stage PD executor for the vLLM backend."""

import json
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from dlrouter.backends.base import PDRequestContext
from dlrouter.backends.vllm.kv_transfer import KVTransferAdapter
from dlrouter.backends.vllm.pair_selection import VLLMPairSelector
from dlrouter.backends.vllm.request_state import VLLMTwoStageRequestState


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
        service_discovery = context.service_discovery
        if service_discovery is None:
            return JSONResponse(
                {'error': 'No service discovery configured for vLLM PD mode'},
                status_code=503,
            )

        pd_pair = self.pair_selector.select_pair(
            prefill_candidates=service_discovery.get_prefill_instances(),
            decode_candidates=service_discovery.get_decode_instances(),
            model_name=request_data.get('model', ''),
        )
        if pd_pair is None:
            return JSONResponse(
                {'error': 'No prefill or decode instances available'},
                status_code=503,
            )

        prefill_info, decode_info = pd_pair
        request_id = self.adapter.build_request_id(prefill_info, decode_info)
        state = VLLMTwoStageRequestState(
            request_id=request_id,
            prefill_url=prefill_info.to_http_url(),
            decode_url=decode_info.to_http_url(),
        )

        prefill_request = self.adapter.build_prefill_request(
            request_data,
            request_id,
            state.aborted_request_ids,
        )
        prefill_text = await self.backend.forward_with_request_id(
            state.prefill_url,
            endpoint,
            prefill_request,
            request_id,
        )
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
                self._stream_decode(state, endpoint, decode_request),
                media_type='text/event-stream',
            )

        text = await self.backend.forward_with_request_id(
            state.decode_url,
            endpoint,
            decode_request,
            request_id,
        )
        return JSONResponse(json.loads(text))

    async def _stream_decode(
        self,
        state: VLLMTwoStageRequestState,
        endpoint: str,
        decode_request: dict[str, Any],
    ):
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
            raise
