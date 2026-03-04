"""Health check for backend nodes."""

import asyncio
import threading
import time
from collections import defaultdict

from dlrouter.constants import (
    HEALTH_CHECK_MAX_FAILURES,
    HEARTBEAT_EXPIRATION,
)
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.health')


class HealthChecker:
    """Periodic health checker for backend nodes.

    Runs a background daemon thread that periodically
    checks all registered nodes and removes stale ones.

    A node is only removed after *max_failures* consecutive
    health-check failures, avoiding premature removal
    caused by transient issues (e.g. cache-block GC in
    PD disaggregation mode).
    """

    def __init__(
        self,
        node_manager,
        interval: int = HEARTBEAT_EXPIRATION,
        max_failures: int = HEALTH_CHECK_MAX_FAILURES,
    ) -> None:
        self._manager = node_manager
        self._interval = interval
        self._max_failures = max_failures
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
        logger.info(f'Health checker started (interval={self._interval}s, max_failures={self._max_failures})')

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
        """Check all nodes; remove after consecutive failures."""
        logger.info('Running health check...')
        node_urls = list(self._manager.nodes.keys())
        backend = self._manager.backend

        loop = asyncio.new_event_loop()
        stale = []
        try:
            for url in node_urls:
                try:
                    healthy = loop.run_until_complete(backend.check_health(url))
                except Exception:
                    healthy = False

                if healthy:
                    # Reset counter on success
                    self._fail_counts[url] = 0
                else:
                    self._fail_counts[url] += 1
                    cnt = self._fail_counts[url]
                    logger.warning(f'Health check failed for {url} ({cnt}/{self._max_failures})')
                    if cnt >= self._max_failures:
                        stale.append(url)
        finally:
            loop.close()

        for url in stale:
            self._manager.remove(url)
            self._fail_counts.pop(url, None)
            logger.info(f'Removed stale node: {url} (failed {self._max_failures} consecutive checks)')
