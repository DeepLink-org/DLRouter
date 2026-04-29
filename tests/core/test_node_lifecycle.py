"""Tests for shared node lifecycle accounting."""

from unittest.mock import MagicMock

from dlrouter.core.node_lifecycle import post_call, pre_call


def test_pre_call_and_post_call_track_node_lifecycle() -> None:
    node_manager = MagicMock()
    node_manager.pre_call.return_value = 123.0

    start = pre_call(node_manager, 'http://node:8000')
    post_call(node_manager, 'http://node:8000', start)

    assert start == 123.0
    node_manager.pre_call.assert_called_once_with('http://node:8000')
    node_manager.post_call.assert_called_once_with('http://node:8000', 123.0)


def test_lifecycle_helpers_ignore_missing_node_manager_capabilities() -> None:
    node_manager = MagicMock()
    node_manager.pre_call.side_effect = KeyError('missing node')

    start = pre_call(node_manager, 'http://missing:8000')
    post_call(node_manager, 'http://missing:8000', start)

    assert start is None
    node_manager.post_call.assert_not_called()
