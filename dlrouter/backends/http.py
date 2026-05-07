"""Shared async HTTP transport helpers for backend adapters."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING, Any

import aiohttp

from dlrouter.constants import AIOHTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.backends.http')

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class StreamFraming(str, Enum):
    """Streaming response framing policy for backend adapters."""

    SSE_LINES = 'sse_lines'
    PASSTHROUGH = 'passthrough'


class BackendHTTPTransportMixin:
    """Reusable async HTTP transport behavior for backend adapters.

    This mixin intentionally does not implement fetch_models(). The current
    BaseBackend contract keeps model fetching synchronous because node
    registration and lazy health-check discovery call it synchronously.

    _create_session() is a protected testing seam, not a public backend
    extension API.
    """

    stream_framing: StreamFraming = StreamFraming.SSE_LINES
    _timeout: aiohttp.ClientTimeout
    _health_timeout: aiohttp.ClientTimeout
    _connector_kwargs: dict[str, Any] | None
    _session: aiohttp.ClientSession | None
    _session_lock: asyncio.Lock | None

    async def _create_session(self) -> aiohttp.ClientSession:
        connector = None
        if self._connector_kwargs:
            connector = aiohttp.TCPConnector(**self._connector_kwargs)
        return aiohttp.ClientSession(
            connector=connector,
            timeout=self._timeout,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a persistent aiohttp session."""
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()

        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = await self._create_session()
        return self._session

    async def close(self) -> None:
        """Close the persistent aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward a non-stream request to a backend node."""
        session = await self._get_session()
        url = node_url + endpoint
        try:
            async with session.post(
                url,
                json=request_data,
                timeout=self._timeout,
            ) as resp:
                return await resp.text()
        except Exception as exc:
            logger.error(f'Backend forward error: {exc}')
            raise

    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream-forward a request to a backend node."""
        session = await self._get_session()
        url = node_url + endpoint
        try:
            async with session.post(
                url,
                json=request_data,
                timeout=self._timeout,
            ) as resp:
                async for chunk in resp.content:
                    framed = self._frame_stream_chunk(chunk)
                    if framed is not None:
                        yield framed
        except Exception as exc:
            logger.error(f'Backend stream error: {exc}')
            raise

    def _frame_stream_chunk(self, chunk: bytes) -> bytes | None:
        """Apply backend stream framing while preserving current behavior."""
        if self.stream_framing == StreamFraming.PASSTHROUGH:
            # Preserve the backend's existing aiohttp content-iterator shape
            # without adding SSE framing. This is intentionally not iter_any().
            return chunk if chunk else None

        # Preserve current vLLM/LMDeploy behavior exactly: append an SSE
        # separator to the yielded line without stripping existing newlines.
        if chunk.strip():
            return chunk + b'\n\n'
        return None

    async def check_health(self, node_url: str) -> bool:
        """Check backend node health without reusing request-loop sessions.

        HealthChecker runs in a background thread with its own event loops, so
        this intentionally uses a short-lived session to avoid cross-loop
        ClientSession reuse after normal forwarding has created a session.
        """
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(
                    f'{node_url}/health',
                    timeout=self._health_timeout,
                ) as resp,
            ):
                return resp.status == 200
        except Exception as exc:
            logger.error(f'Failed to check health from {node_url}: {exc}')
            return False


def default_timeout() -> aiohttp.ClientTimeout:
    """Return the default backend request timeout."""
    return aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)


def default_health_timeout() -> aiohttp.ClientTimeout:
    """Return the default backend health-check timeout."""
    return aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
