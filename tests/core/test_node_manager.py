"""Tests for NodeManager node lifecycle behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlrouter.core.node_manager import NodeManager
from dlrouter.models.node import NodeStatus


if TYPE_CHECKING:
    import pytest


class DummyBackend:
    def deregister_node(self, node_url: str) -> None:
        return None


class FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


def _node_manager() -> NodeManager:
    return NodeManager(
        backend=DummyBackend(),  # type: ignore[arg-type]
        cache_status=False,
    )


def test_terminate_node_normalizes_dp_aware_url_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _node_manager()
    manager.nodes['http://node:8000@3'] = NodeStatus(models=['qwen'])
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({'url': url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr('dlrouter.core.node_manager.requests.get', fake_get)

    assert manager.terminate_node('http://node:8000@3') is True

    assert calls == [
        {
            'url': 'http://node:8000/terminate',
            'headers': {'accept': 'application/json'},
        },
    ]
    assert manager.nodes == {}


def test_terminate_all_terminates_each_dp_aware_base_url_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _node_manager()
    manager.nodes['http://node-a:8000@0'] = NodeStatus(models=['qwen'])
    manager.nodes['http://node-a:8000@1'] = NodeStatus(models=['qwen'])
    manager.nodes['http://node-b:8000'] = NodeStatus(models=['qwen'])
    calls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr('dlrouter.core.node_manager.requests.get', fake_get)

    assert manager.terminate_all() is True

    assert calls == [
        'http://node-a:8000/terminate',
        'http://node-b:8000/terminate',
    ]
    assert manager.nodes == {}
