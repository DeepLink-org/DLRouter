"""Tests for vLLM pair selection compatibility rules."""

from dlrouter.backends.vllm.pair_selection import VLLMPairSelector
from dlrouter.constants import EngineRole
from dlrouter.core.service_discovery import NodeInfo


def _node(
    http_address: str,
    role: EngineRole,
    *,
    models: list[str] | None = None,
    kv_connector: str = 'mooncake',
    protocol_version: str = 'v1',
    endpoint_metadata: dict[str, str] | None = None,
) -> NodeInfo:
    return NodeInfo(
        http_address=http_address,
        role=role,
        models=['qwen3-32b'] if models is None else models,
        metadata={
            'kv_connector': kv_connector,
            'protocol_version': protocol_version,
            'endpoint_metadata': endpoint_metadata or {'transport': 'rdma'},
        },
    )


def test_filter_pairs_keeps_only_minimally_compatible_nodes() -> None:
    selector = VLLMPairSelector()
    prefill_candidates = [
        _node('10.0.0.1:13700', EngineRole.PREFILL),
        _node('10.0.0.3:13700', EngineRole.PREFILL, kv_connector='nixl'),
    ]
    decode_candidates = [
        _node('10.0.0.2:13701', EngineRole.DECODE),
        _node('10.0.0.4:13701', EngineRole.DECODE, protocol_version='v2'),
        _node(
            '10.0.0.5:13701',
            EngineRole.DECODE,
            endpoint_metadata={'transport': 'tcp'},
        ),
    ]

    pairs = selector.filter_pairs(
        prefill_candidates=prefill_candidates,
        decode_candidates=decode_candidates,
        model_name='qwen3-32b',
    )

    assert [(prefill.http_address, decode.http_address) for prefill, decode in pairs] == [
        ('10.0.0.1:13700', '10.0.0.2:13701')
    ]


def test_select_pair_returns_none_when_no_compatible_pair_exists() -> None:
    selector = VLLMPairSelector()
    pair = selector.select_pair(
        prefill_candidates=[_node('10.0.0.1:13700', EngineRole.PREFILL, models=['qwen3-32b'])],
        decode_candidates=[_node('10.0.0.2:13701', EngineRole.DECODE, models=['llama3'])],
        model_name='qwen3-32b',
    )

    assert pair is None


def test_select_pair_rejects_empty_model_lists() -> None:
    selector = VLLMPairSelector()
    pair = selector.select_pair(
        prefill_candidates=[
            _node(
                '10.0.0.1:13700',
                EngineRole.PREFILL,
                models=[],
            )
        ],
        decode_candidates=[
            _node(
                '10.0.0.2:13701',
                EngineRole.DECODE,
                models=[],
            )
        ],
        model_name='qwen3-32b',
    )

    assert pair is None


def test_select_pair_advances_round_robin_across_compatible_pairs() -> None:
    selector = VLLMPairSelector()
    prefill_candidates = [
        _node('10.0.0.1:13700', EngineRole.PREFILL),
        _node('10.0.0.2:13700', EngineRole.PREFILL),
    ]
    decode_candidates = [
        _node('10.0.0.1:13701', EngineRole.DECODE),
        _node('10.0.0.2:13701', EngineRole.DECODE),
    ]

    first_pair = selector.select_pair(
        prefill_candidates=prefill_candidates,
        decode_candidates=decode_candidates,
        model_name='qwen3-32b',
    )
    second_pair = selector.select_pair(
        prefill_candidates=prefill_candidates,
        decode_candidates=decode_candidates,
        model_name='qwen3-32b',
    )

    assert first_pair is not None
    assert second_pair is not None
    assert first_pair != second_pair
