"""Pair selection for SGLang PD dual dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dlrouter.constants import EngineRole


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


class SGLangPairSelector:
    """Select a prefill/decode pair from NodeManager."""

    def select_pair(
        self,
        *,
        node_manager: NodeManager,
        model_name: str,
        request_key: str | None = None,
    ) -> tuple[str, str] | None:
        """Select a P/D pair for SGLang PD proxying."""
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

        return prefill_url, decode_url
