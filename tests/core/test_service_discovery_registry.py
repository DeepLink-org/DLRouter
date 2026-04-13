"""Tests for the unified service discovery registry."""

from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo, ServiceDiscoveryRegistry


def test_registry_supports_upsert_remove_and_role_views() -> None:
    registry = ServiceDiscoveryRegistry()
    prefill = NodeInfo(
        http_address='10.0.0.1:8000',
        role=EngineRole.PREFILL,
        models=['qwen3-32b'],
        metadata={'kv_connector': 'mooncake', 'protocol_version': 'v1'},
    )
    decode = NodeInfo(
        http_address='10.0.0.2:8000',
        role=EngineRole.DECODE,
        models=['qwen3-32b'],
        metadata={'kv_connector': 'mooncake', 'protocol_version': 'v1'},
    )

    registry.upsert(prefill)
    registry.upsert(decode)

    assert [node.http_address for node in registry.get_prefill_instances()] == ['10.0.0.1:8000']
    assert [node.http_address for node in registry.get_decode_instances()] == ['10.0.0.2:8000']
    assert {node.http_address for node in registry.get_all_instances()} == {
        '10.0.0.1:8000',
        '10.0.0.2:8000',
    }

    removed = registry.remove(prefill)

    assert removed == prefill
    assert registry.get_prefill_instances() == []
    assert [node.http_address for node in registry.get_decode_instances()] == ['10.0.0.2:8000']


def test_registry_filters_by_model_and_metadata_subset() -> None:
    registry = ServiceDiscoveryRegistry()
    registry.upsert(
        NodeInfo(
            http_address='10.0.0.1:8000',
            role=EngineRole.PREFILL,
            models=['qwen3-32b'],
            metadata={
                'kv_connector': 'mooncake',
                'protocol_version': 'v1',
                'transport': 'rdma',
            },
        )
    )
    registry.upsert(
        NodeInfo(
            http_address='10.0.0.2:8000',
            role=EngineRole.PREFILL,
            models=['llama3'],
            metadata={'kv_connector': 'nixl', 'protocol_version': 'v1'},
        )
    )

    result = registry.filter_instances(
        role=EngineRole.PREFILL,
        model='qwen3-32b',
        metadata={'kv_connector': 'mooncake', 'protocol_version': 'v1'},
    )

    assert [node.http_address for node in result] == ['10.0.0.1:8000']


def test_registry_key_includes_role_so_same_address_different_roles_coexist() -> None:
    registry = ServiceDiscoveryRegistry()
    prefill = NodeInfo(
        http_address='10.0.0.1:8000',
        role=EngineRole.PREFILL,
        models=['qwen3-32b'],
        metadata={'kv_connector': 'mooncake'},
    )
    decode = NodeInfo(
        http_address='10.0.0.1:8000',
        role=EngineRole.DECODE,
        models=['qwen3-32b'],
        metadata={'kv_connector': 'mooncake'},
    )

    registry.upsert(prefill)
    registry.upsert(decode)

    assert len(registry.get_all_instances()) == 2
    assert len(registry.get_prefill_instances()) == 1
    assert len(registry.get_decode_instances()) == 1


def test_registry_upsert_overwrites_same_role_and_address() -> None:
    registry = ServiceDiscoveryRegistry()
    original = NodeInfo(
        http_address='10.0.0.1:8000',
        role=EngineRole.PREFILL,
        models=['qwen3-32b'],
        metadata={'kv_connector': 'mooncake', 'protocol_version': 'v1'},
    )
    updated = NodeInfo(
        http_address='10.0.0.1:8000',
        role=EngineRole.PREFILL,
        models=['qwen3-32b'],
        metadata={
            'kv_connector': 'mooncake',
            'protocol_version': 'v1',
            'generation': 'new',
        },
    )

    registry.upsert(original)
    registry.upsert(updated)

    instances = registry.get_prefill_instances()

    assert len(instances) == 1
    assert instances[0].metadata.get('generation') == 'new'
    assert registry.remove(updated) == updated
    assert registry.get_all_instances() == []
