"""Shared contract tests for builtin backends."""

import pytest

from dlrouter.backends.factory import create_backend, get_backend_definition
from dlrouter.constants import BackendType


@pytest.mark.parametrize(
    ('backend_type', 'expected_name'),
    [
        (BackendType.VLLM, 'vllm'),
        (BackendType.LMDEPLOY, 'lmdeploy'),
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

    assert hasattr(backend, 'forward_request')
    assert hasattr(backend, 'stream_forward')
    assert hasattr(backend, 'fetch_models')
    assert hasattr(backend, 'check_health')
