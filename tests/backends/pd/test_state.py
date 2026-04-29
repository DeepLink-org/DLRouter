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


def test_two_stage_state_only_tracks_values_read_by_the_executor() -> None:
    state = TwoStageRequestState(
        request_id='req-1',
        prefill_url='http://prefill:8000',
        decode_url='http://decode:8000',
    )

    assert not hasattr(state, 'prefill_kv_released')
