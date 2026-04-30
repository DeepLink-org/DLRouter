"""Safe node lifecycle accounting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager


logger = get_logger('dlrouter.core.node_lifecycle')


def pre_call(node_manager: NodeManager, node_url: str) -> float | None:
    """Track request start on a node when the manager supports it."""
    try:
        return node_manager.pre_call(node_url)
    except Exception as exc:
        logger.debug(f'pre_call skipped for {node_url}: {exc}')
        return None


def post_call(node_manager: NodeManager, node_url: str, start: float | None) -> None:
    """Track request completion on a node when a matching start exists."""
    if start is None:
        return
    try:
        node_manager.post_call(node_url, start)
    except Exception as exc:
        logger.debug(f'post_call skipped for {node_url}: {exc}')
