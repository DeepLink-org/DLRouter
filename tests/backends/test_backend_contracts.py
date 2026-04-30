"""Shared contract tests for builtin backends."""

import pytest

from dlrouter.backends.factory import create_backend, get_backend_definition
from dlrouter.constants import BackendType, ServiceDiscoveryMode


@pytest.mark.parametrize(
    ('backend_type', 'expected_name'),
    [
        (BackendType.VLLM, 'vllm'),
        (BackendType.LMDEPLOY, 'lmdeploy'),
        (BackendType.SGLANG, 'sglang'),
    ],
)
def test_builtin_backends_expose_phase_one_capabilities(
    backend_type: BackendType,
    expected_name: str,
):
    definition = get_backend_definition(backend_type)
    backend = create_backend(backend_type)

    assert definition.name == expected_name
    assert definition.supports('forward_request') is True
    assert definition.supports('stream_forward') is True
    assert definition.supports('fetch_models') is True
    assert definition.supports('check_health') is True
    assert definition.supports('register_node') is True
    assert definition.supports('deregister_node') is True
    assert definition.supports('handle_pd_request') is True

    assert hasattr(backend, 'forward_request')
    assert hasattr(backend, 'stream_forward')
    assert hasattr(backend, 'fetch_models')
    assert hasattr(backend, 'check_health')
    assert hasattr(backend, 'handle_pd_request')
    assert backend.supports_pd_disagg() is True


@pytest.mark.parametrize(
    ('backend_type', 'backend_config', 'expected_mode'),
    [
        (BackendType.VLLM, {}, ServiceDiscoveryMode.HEARTBEAT),
        (
            BackendType.VLLM,
            {
                'prefill_urls': 'http://10.0.0.1:8200',
                'decode_urls': 'http://10.0.0.2:8200',
            },
            ServiceDiscoveryMode.STATIC,
        ),
        (BackendType.SGLANG, {}, ServiceDiscoveryMode.STATIC),
        (BackendType.LMDEPLOY, {}, None),
    ],
)
def test_builtin_backends_return_expected_discovery_preference(
    backend_type: BackendType,
    backend_config: dict[str, str],
    expected_mode: ServiceDiscoveryMode | None,
):
    backend = create_backend(backend_type)

    assert backend.preferred_discovery_mode(backend_config) is expected_mode
