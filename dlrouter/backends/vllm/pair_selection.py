"""Pair selection for vLLM PD — fully based on NodeManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.pair_selection')


class VLLMPairSelector:
    """Select prefill/decode pairs for vLLM two-stage PD.

    Uses NodeManager as the single source of truth for node candidates
    and delegates routing to NodeManager's configured routing strategy.
    """

    def select_pair(
        self,
        *,
        node_manager: NodeManager,
        model_name: str,
    ) -> tuple[str, str] | None:
        """Select a P/D pair.

        Returns (prefill_url, decode_url) or None if no candidates.
        """
        prefill_candidates = {url: st for url, st in node_manager.prefill_nodes.items() if model_name in st.models}
        decode_candidates = {url: st for url, st in node_manager.decode_nodes.items() if model_name in st.models}
        if not prefill_candidates or not decode_candidates:
            return None

        prefill_url = node_manager._router.select_node(model_name, prefill_candidates)
        decode_url = node_manager._router.select_node(model_name, decode_candidates)
        if prefill_url is None or decode_url is None:
            return None

        return prefill_url, decode_url
