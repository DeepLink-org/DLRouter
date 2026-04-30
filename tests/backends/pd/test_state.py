"""Tests for shared PD request state."""

from dlrouter.backends.pd.state import TwoStageRequestState


def test_two_stage_state_tracks_aborted_request_ids() -> None:
    state = TwoStageRequestState(
        request_id='req-1',
        prefill_url='http://prefill:8000',
        decode_url='http://decode:8000',
    )

    state.mark_aborted()

    assert state.aborted_request_ids == ['req-1']


def test_two_stage_state_keeps_abort_tracking_request_local() -> None:
    first = TwoStageRequestState(
        request_id='req-1',
        prefill_url='http://prefill-1:8000',
        decode_url='http://decode-1:8000',
    )
    second = TwoStageRequestState(
        request_id='req-2',
        prefill_url='http://prefill-2:8000',
        decode_url='http://decode-2:8000',
    )

    first.mark_aborted()

    assert first.aborted_request_ids == ['req-1']
    assert second.aborted_request_ids == []


def test_two_stage_state_starts_each_request_with_empty_abort_tracking() -> None:
    first = TwoStageRequestState(
        request_id='req-1',
        prefill_url='http://prefill-1:8000',
        decode_url='http://decode-1:8000',
    )
    second = TwoStageRequestState(
        request_id='req-2',
        prefill_url='http://prefill-2:8000',
        decode_url='http://decode-2:8000',
    )

    assert first.aborted_request_ids == []
    assert second.aborted_request_ids == []
    assert first.aborted_request_ids is not second.aborted_request_ids


def test_two_stage_state_only_tracks_values_read_by_the_executor() -> None:
    state = TwoStageRequestState(
        request_id='req-1',
        prefill_url='http://prefill:8000',
        decode_url='http://decode:8000',
    )

    assert not hasattr(state, 'prefill_kv_released')
