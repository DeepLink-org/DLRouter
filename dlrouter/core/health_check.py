"""Health check for backend nodes."""

import asyncio
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from dlrouter.constants import (
    HEALTH_CHECK_MAX_FAILURES,
    HEARTBEAT_EXPIRATION,
)
from dlrouter.logger import get_logger


if TYPE_CHECKING:
    from dlrouter.core.node_manager import NodeManager

logger = get_logger('dlrouter.health')


class HealthChecker:
    """Periodic health checker for backend nodes.

    Runs a background daemon thread that periodically
    checks all registered nodes and removes stale ones.

    A node is only removed after *max_failures* consecutive
    health-check failures, avoiding premature removal
    caused by transient issues (e.g. cache-block GC in
    PD disaggregation mode).

    Uses asyncio.gather to check all nodes in parallel,
    significantly reducing total check time.
    """

    def __init__(
        self,
        node_manager: 'NodeManager',
        interval: int = HEARTBEAT_EXPIRATION,
        max_failures: int = HEALTH_CHECK_MAX_FAILURES,
        batch_size: int = 50,
    ) -> None:
        self._manager = node_manager
        self._interval = interval
        self._max_failures = max_failures
        self._batch_size = batch_size  # Max concurrent health checks
        self._thread = None
        self._running = False
        # Track consecutive failures per node URL
        self._fail_counts: dict[str, int] = defaultdict(int)

    def start(self) -> None:
        """Start the health check loop."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name='dlrouter-health',
        )
        self._thread.start()
        logger.info(
            f'Health checker started '
            f'(interval={self._interval}s, '
            f'max_failures={self._max_failures}, '
            f'batch_size={self._batch_size})',
        )

    def stop(self) -> None:
        """Stop the health check loop."""
        self._running = False
        self._thread = None

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            try:
                self._check()
            except Exception as e:
                logger.error(f'Health check error: {e}')

    def _check(self) -> None:
        """Check all nodes in parallel; remove after consecutive failures."""
        node_urls = list(self._manager.nodes.keys())
        if not node_urls:
            return
        logger.info(f'start running health check, {node_urls=}')
        start_time = time.time()

        backend = self._manager.backend
        loop = asyncio.new_event_loop()

        try:
            # Check all nodes in parallel batches
            results = loop.run_until_complete(
                self._check_nodes_batch(backend, node_urls),
            )
        finally:
            loop.close()

        # Process results
        stale = []
        for url, healthy in results:
            if healthy:
                self._fail_counts[url] = 0
            else:
                self._fail_counts[url] += 1
                cnt = self._fail_counts[url]
                logger.warning(
                    f'Health check failed for {url} ({cnt}/{self._max_failures})',
                )
                if cnt >= self._max_failures:
                    stale.append(url)

        # Remove stale nodes
        for url in stale:
            self._manager.remove(url)
            self._fail_counts.pop(url, None)
            logger.info(
                f'Removed stale node: {url} (failed {self._max_failures} consecutive checks)',
            )
        end_time = time.time()
        logger.info(f'finish running health check, {end_time - start_time:.2f}s elapsed')

    async def _check_nodes_batch(
        self,
        backend,
        node_urls: list[str],
    ) -> list[tuple[str, bool]]:
        """Check multiple nodes in parallel with batching.

        Uses asyncio.gather with semaphore to limit concurrency.
        """
        semaphore = asyncio.Semaphore(self._batch_size)

        async def check_one(url: str) -> tuple[str, bool]:
            async with semaphore:
                try:
                    healthy = await backend.check_health(url)
                    return (url, healthy)
                except Exception:
                    return (url, False)

        tasks = [check_one(url) for url in node_urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results
