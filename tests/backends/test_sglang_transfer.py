"""Tests for SGLang bootstrap request injection."""

from dlrouter.backends.sglang.transfer import SGLangBootstrapAdapter


def test_build_request_injects_single_bootstrap_fields_without_mutating_input():
    adapter = SGLangBootstrapAdapter(
        {'http://10.0.0.1:8100': 8998},
        room_generator=lambda: 12345,
    )
    original = {'model': 'qwen3-32b', 'messages': []}

    request = adapter.build_request(
        original,
        prefill_url='http://10.0.0.1:8100',
        endpoint='/v1/chat/completions',
    )

    assert original == {'model': 'qwen3-32b', 'messages': []}
    assert request['bootstrap_host'] == '10.0.0.1'
    assert request['bootstrap_port'] == 8998
    assert request['bootstrap_room'] == 12345


def test_build_request_injects_completion_batch_bootstrap_arrays():
    rooms = iter([11, 22])
    adapter = SGLangBootstrapAdapter(
        {'http://10.0.0.1:8100': 8999},
        room_generator=lambda: next(rooms),
    )

    request = adapter.build_request(
        {'model': 'qwen3-32b', 'prompt': ['hello', 'world']},
        prefill_url='http://10.0.0.1:8100',
        endpoint='/v1/completions',
    )

    assert request['bootstrap_host'] == ['10.0.0.1', '10.0.0.1']
    assert request['bootstrap_port'] == [8999, 8999]
    assert request['bootstrap_room'] == [11, 22]


def test_build_request_injects_chat_n_batch_bootstrap_arrays():
    rooms = iter([101, 202])
    adapter = SGLangBootstrapAdapter(
        {'http://10.0.0.1:8100': 8998},
        room_generator=lambda: next(rooms),
    )

    request = adapter.build_request(
        {'model': 'qwen3-32b', 'messages': [], 'n': 2},
        prefill_url='http://10.0.0.1:8100',
        endpoint='/v1/chat/completions',
    )

    assert request['bootstrap_host'] == ['10.0.0.1', '10.0.0.1']
    assert request['bootstrap_port'] == [8998, 8998]
    assert request['bootstrap_room'] == [101, 202]


def test_build_request_wraps_ipv6_bootstrap_host():
    adapter = SGLangBootstrapAdapter(
        {'http://[2001:db8::1]:8100': 8998},
        room_generator=lambda: 1,
    )

    request = adapter.build_request(
        {'model': 'qwen3-32b', 'messages': []},
        prefill_url='http://[2001:db8::1]:8100',
        endpoint='/v1/chat/completions',
    )

    assert request['bootstrap_host'] == '[2001:db8::1]'
