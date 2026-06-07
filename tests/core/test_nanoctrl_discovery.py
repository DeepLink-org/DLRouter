"""Tests for dlslime-ctrl based NanoDeploy discovery."""

from unittest.mock import MagicMock

import pytest

from dlrouter.backends.factory import create_backend
from dlrouter.backends.nanodeploy.backend import _sanitize_chat_payload
from dlrouter.constants import BackendType, EngineRole, ServiceDiscoveryMode
from dlrouter.core.node_manager import NodeManager
from dlrouter.core.service_discovery.nanoctrl_discovery import (
    NanoCtrlServiceDiscovery,
    _entity_http_url,
    _entity_models,
    _entity_role,
)


def test_entity_url_and_models_helpers():
    entity = {
        'endpoint': {'host': '10.0.0.1', 'port': 8100, 'protocol': 'http'},
        'metadata': {
            'served_model_name': 'Qwen3-0.6B',
            'model_path': '/home/jimy/models/Qwen3-0.6B',
        },
    }
    assert _entity_http_url(entity) == 'http://10.0.0.1:8100'
    assert _entity_models(entity) == [
        'Qwen3-0.6B',
        '/home/jimy/models/Qwen3-0.6B',
    ]


def test_sanitize_chat_payload_strips_router_fields():
    raw = {
        'model': 'Qwen3-0.6B',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'stream': False,
        'top_k': 40,
        'session_id': 'abc',
        'session_params': {'session_id': 'abc'},
    }
    assert _sanitize_chat_payload(raw) == {
        'model': 'Qwen3-0.6B',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'stream': False,
    }


def test_entity_role_maps_metadata_role():
    assert _entity_role({}) == EngineRole.HYBRID
    assert _entity_role({'metadata': {'role': 'hybrid'}}) == EngineRole.HYBRID
    assert _entity_role({'metadata': {'role': 'prefill'}}) == EngineRole.PREFILL
    assert _entity_role({'metadata': {'role': 'Decode'}}) == EngineRole.DECODE
    assert _entity_role({'metadata': {'role': 'bogus'}}) == EngineRole.HYBRID


def test_nanoctrl_discovery_assigns_pd_roles():
    backend = create_backend(BackendType.NANODEPLOY)
    node_manager = NodeManager(backend=backend)

    discovery = NanoCtrlServiceDiscovery(
        ctrl_address='127.0.0.1:4479',
        node_manager=node_manager,
        ctrl_kind='nanodeploy',
        poll_interval=60.0,
    )

    fake_client = MagicMock()
    fake_client.list_entities.return_value = [
        {
            'endpoint': {'host': '127.0.0.1', 'port': 8100, 'protocol': 'http'},
            'metadata': {'served_model_name': 'Qwen3-4B', 'role': 'prefill'},
        },
        {
            'endpoint': {'host': '127.0.0.1', 'port': 8200, 'protocol': 'http'},
            'metadata': {'served_model_name': 'Qwen3-4B', 'role': 'decode'},
        },
    ]
    discovery._client = fake_client

    discovery._poll_once()
    assert node_manager.nodes['http://127.0.0.1:8100'].role == EngineRole.PREFILL
    assert node_manager.nodes['http://127.0.0.1:8200'].role == EngineRole.DECODE


def test_nanodeploy_backend_prefers_nanoctrl_when_ctrl_set():
    backend = create_backend(
        BackendType.NANODEPLOY,
        {'ctrl_address': '127.0.0.1:4479'},
    )
    assert backend.preferred_discovery_mode({'ctrl_address': '127.0.0.1:4479'}) == (
        ServiceDiscoveryMode.NANOCTRL
    )
    assert backend.preferred_discovery_mode({}) is None
    assert backend.supports_pd_disagg() is True


def test_nanoctrl_discovery_syncs_nodes():
    backend = create_backend(BackendType.NANODEPLOY)
    node_manager = NodeManager(backend=backend)

    discovery = NanoCtrlServiceDiscovery(
        ctrl_address='127.0.0.1:4479',
        node_manager=node_manager,
        ctrl_kind='nanodeploy',
        poll_interval=60.0,
    )

    fake_client = MagicMock()
    fake_client.list_entities.return_value = [
        {
            'endpoint': {'host': '127.0.0.1', 'port': 8100, 'protocol': 'http'},
            'metadata': {'served_model_name': 'Qwen3-4B'},
        },
    ]
    discovery._client = fake_client

    discovery._poll_once()
    assert 'http://127.0.0.1:8100' in node_manager.nodes
    assert node_manager.nodes['http://127.0.0.1:8100'].models == ['Qwen3-4B']

    fake_client.list_entities.return_value = []
    discovery._poll_once()
    assert 'http://127.0.0.1:8100' not in node_manager.nodes
