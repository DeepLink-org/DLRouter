"""Tests for shared backend HTTP transport helpers."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from dlrouter.backends.http import BackendHTTPTransportMixin, StreamFraming


class FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> FakeContent:
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class FakeResponse:
    def __init__(
        self,
        *,
        text: str = 'ok',
        status: int = 200,
        chunks: list[bytes] | None = None,
    ) -> None:
        self._text = text
        self.status = status
        self.content = FakeContent(chunks or [])

    async def text(self) -> str:
        return self._text


class FakeRequestContext:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    async def __aenter__(self) -> FakeResponse:
        return self.response

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.closed = False
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self.next_post_response = FakeResponse(text='posted')
        self.next_get_response = FakeResponse(status=200)

    def post(self, url: str, **kwargs: Any) -> FakeRequestContext:
        self.posts.append({'url': url, **kwargs})
        return FakeRequestContext(self.next_post_response)

    def get(self, url: str, **kwargs: Any) -> FakeRequestContext:
        self.gets.append({'url': url, **kwargs})
        return FakeRequestContext(self.next_get_response)

    async def close(self) -> None:
        self.closed = True


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class ExampleBackend(BackendHTTPTransportMixin):
    stream_framing = StreamFraming.SSE_LINES

    def __init__(self) -> None:
        self._timeout = aiohttp.ClientTimeout(total=5)
        self._health_timeout = aiohttp.ClientTimeout(total=1)
        self._connector_kwargs: dict[str, Any] | None = None
        self._session: FakeSession | None = None
        self._session_lock = None
        self._test_session = FakeSession()

    async def _create_session(self) -> FakeSession:
        return self._test_session


@pytest.mark.asyncio
async def test_get_session_reuses_open_session() -> None:
    backend = ExampleBackend()

    first = await backend._get_session()
    second = await backend._get_session()

    assert first is second


@pytest.mark.asyncio
async def test_close_closes_and_clears_session() -> None:
    backend = ExampleBackend()
    session = await backend._get_session()

    await backend.close()

    assert session.closed is True
    assert backend._session is None


@pytest.mark.asyncio
async def test_forward_request_posts_json_and_returns_text() -> None:
    backend = ExampleBackend()

    result = await backend.forward_request(
        'http://node:8000',
        '/v1/chat/completions',
        {'model': 'qwen'},
    )

    assert result == 'posted'
    assert backend._test_session.posts == [
        {
            'url': 'http://node:8000/v1/chat/completions',
            'json': {'model': 'qwen'},
            'timeout': backend._timeout,
        },
    ]


@pytest.mark.asyncio
async def test_forward_request_normalizes_dp_aware_url_before_post() -> None:
    backend = ExampleBackend()

    await backend.forward_request(
        'http://node:8000@3',
        '/v1/chat/completions',
        {'model': 'qwen'},
    )

    assert backend._test_session.posts[0]['url'] == 'http://node:8000/v1/chat/completions'


@pytest.mark.asyncio
async def test_stream_forward_sse_lines_skips_blank_lines_and_appends_separator() -> None:
    backend = ExampleBackend()
    backend._test_session.next_post_response = FakeResponse(
        chunks=[b'data: one\n', b'\n', b'data: two\n'],
    )

    chunks = [
        chunk
        async for chunk in backend.stream_forward(
            'http://node:8000',
            '/v1/chat/completions',
            {'stream': True},
        )
    ]

    # Preserve current vLLM/LMDeploy behavior: append b'\n\n' without
    # stripping an existing newline from the upstream line.
    assert chunks == [b'data: one\n\n\n', b'data: two\n\n\n']


@pytest.mark.asyncio
async def test_stream_forward_normalizes_dp_aware_url_before_post() -> None:
    backend = ExampleBackend()
    backend._test_session.next_post_response = FakeResponse(chunks=[b'data: one\n'])

    chunks = [
        chunk
        async for chunk in backend.stream_forward(
            'http://node:8000@3',
            '/v1/chat/completions',
            {'stream': True},
        )
    ]

    assert chunks == [b'data: one\n\n\n']
    assert backend._test_session.posts[0]['url'] == 'http://node:8000/v1/chat/completions'


@pytest.mark.asyncio
async def test_stream_forward_passthrough_keeps_content_iterator_shape() -> None:
    class PassthroughBackend(ExampleBackend):
        stream_framing = StreamFraming.PASSTHROUGH

    backend = PassthroughBackend()
    backend._test_session.next_post_response = FakeResponse(
        chunks=[b'chunk-one', b'', b'chunk-two'],
    )

    chunks = [
        chunk
        async for chunk in backend.stream_forward(
            'http://node:8000',
            '/generate',
            {'stream': True},
        )
    ]

    assert chunks == [b'chunk-one', b'chunk-two']


@pytest.mark.asyncio
async def test_check_health_uses_short_lived_session_and_health_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ExampleBackend()
    health_session = FakeSession()
    monkeypatch.setattr(
        aiohttp,
        'ClientSession',
        lambda: FakeSessionContext(health_session),
    )

    result = await backend.check_health('http://node:8000')

    assert result is True
    assert health_session.gets == [
        {
            'url': 'http://node:8000/health',
            'timeout': backend._health_timeout,
        },
    ]


@pytest.mark.asyncio
async def test_check_health_does_not_reuse_forwarding_session(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ExampleBackend()
    forwarding_session = await backend._get_session()
    health_session = FakeSession()

    def fail_if_forwarding_session_get_is_used(url: str, **kwargs: Any) -> FakeRequestContext:
        raise AssertionError('health check must not reuse the forwarding session')

    forwarding_session.get = fail_if_forwarding_session_get_is_used
    monkeypatch.setattr(
        aiohttp,
        'ClientSession',
        lambda: FakeSessionContext(health_session),
    )

    assert await backend.check_health('http://node:8000') is True
    assert health_session.gets == [
        {
            'url': 'http://node:8000/health',
            'timeout': backend._health_timeout,
        },
    ]


@pytest.mark.asyncio
async def test_check_health_returns_false_for_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ExampleBackend()
    health_session = FakeSession()
    health_session.next_get_response = FakeResponse(status=503)
    monkeypatch.setattr(
        aiohttp,
        'ClientSession',
        lambda: FakeSessionContext(health_session),
    )

    assert await backend.check_health('http://node:8000') is False
