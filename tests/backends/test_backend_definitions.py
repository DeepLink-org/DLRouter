"""Tests for backend definitions and registry helpers."""

import pytest

from dlrouter.backends.definition import BackendDefinition
from dlrouter.backends.factory import get_backend_definition
from dlrouter.backends.lmdeploy import LMDEPLOY_BACKEND_DEFINITION
from dlrouter.backends.sglang import SGLANG_BACKEND_DEFINITION
from dlrouter.backends.vllm import VLLM_BACKEND_DEFINITION
from dlrouter.constants import BackendType


def test_get_backend_definition_returns_vllm_definition():
    definition = get_backend_definition(BackendType.VLLM)

    assert isinstance(definition, BackendDefinition)
    assert definition.backend_type is BackendType.VLLM
    assert definition.name == 'vllm'
    assert definition.supports('forward_request') is True


def test_get_backend_definition_returns_lmdeploy_definition():
    definition = get_backend_definition(BackendType.LMDEPLOY)

    assert definition is LMDEPLOY_BACKEND_DEFINITION


def test_definition_exports_are_shared_singletons():
    assert get_backend_definition(BackendType.VLLM) is VLLM_BACKEND_DEFINITION


def test_builtin_definitions_do_not_repeat_base_backend_capabilities():
    assert VLLM_BACKEND_DEFINITION.capability_names == ()
    assert LMDEPLOY_BACKEND_DEFINITION.capability_names == ()
    assert SGLANG_BACKEND_DEFINITION.capability_names == ()


def test_get_backend_definition_raises_for_unknown_backend():
    with pytest.raises(ValueError, match='Unsupported backend'):
        get_backend_definition('unknown')  # type: ignore[arg-type]
