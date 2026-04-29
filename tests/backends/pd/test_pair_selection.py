"""Tests for shared PD pair selection."""

from unittest.mock import MagicMock

from dlrouter.backends.pd import PDPair, PDPairSelector


def test_select_pair_returns_pd_pair_when_prefill_and_decode_exist() -> None:
    node_manager = MagicMock()
    node_manager.get_node_url.side_effect = [
        'http://prefill:8000',
        'http://decode:8000',
    ]
    selector = PDPairSelector()

    pair = selector.select_pair(
        node_manager=node_manager,
        model_name='qwen3-32b',
        request_key='prompt-prefix',
    )

    assert pair == PDPair(
        prefill_url='http://prefill:8000',
        decode_url='http://decode:8000',
    )
    assert node_manager.get_node_url.call_args_list[0].kwargs['request_key'] == 'prompt-prefix'
    assert node_manager.get_node_url.call_args_list[1].kwargs['request_key'] == 'prompt-prefix'


def test_select_pair_returns_none_when_either_role_is_missing() -> None:
    node_manager = MagicMock()
    node_manager.get_node_url.side_effect = ['http://prefill:8000', None]
    selector = PDPairSelector()

    pair = selector.select_pair(
        node_manager=node_manager,
        model_name='qwen3-32b',
    )

    assert pair is None
