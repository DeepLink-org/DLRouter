"""Safe node lifecycle accounting helpers."""

from typing import Any, Optional

from dlrouter.logger import get_logger


logger = get_logger('dlrouter.core.node_lifecycle')


def pre_call(node_manager: Any, node_url: str) -> Optional[float]:
    """Track request start on a node when the manager supports it."""
    try:
        return node_manager.pre_call(node_url)
    except Exception as exc:
        logger.debug(f'pre_call skipped for {node_url}: {exc}')
        return None


def post_call(node_manager: Any, node_url: str, start: Optional[float]) -> None:
    """Track request completion on a node when a matching start exists."""
    if start is None:
        return
    try:
        node_manager.post_call(node_url, start)
    except Exception as exc:
        logger.debug(f'post_call skipped for {node_url}: {exc}')
