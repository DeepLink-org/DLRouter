"""Shared Prefill/Decode pair selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from dlrouter.constants import EngineRole


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


@dataclass(frozen=True)
class PDPair:
    """Selected prefill/decode node pair."""

    prefill_url: str
    decode_url: str


class PDPairSelector:
    """Select a prefill/decode pair through NodeManager routing."""

    def select_pair(
        self,
        *,
        node_manager: NodeManager,
        model_name: str,
        request_key: str | None = None,
    ) -> PDPair | None:
        """Return a selected P/D pair, or None when either role is unavailable."""
        prefill_url = node_manager.get_node_url(
            model_name,
            role=EngineRole.PREFILL,
            request_key=request_key,
        )
        decode_url = node_manager.get_node_url(
            model_name,
            role=EngineRole.DECODE,
            request_key=request_key,
        )
        if prefill_url is None or decode_url is None:
            return None

        return PDPair(prefill_url=prefill_url, decode_url=decode_url)


def no_pd_pair_response() -> JSONResponse:
    """Return the existing error shape for unavailable P/D pairs."""
    return JSONResponse(
        {'error': 'No prefill or decode instances available'},
        status_code=503,
    )
