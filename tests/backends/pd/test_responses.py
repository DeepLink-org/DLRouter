"""Tests for shared PD responses."""

import json

from dlrouter.backends.pd.selection import no_pd_pair_response


def test_no_pd_pair_response_matches_existing_executor_error_shape() -> None:
    response = no_pd_pair_response()

    assert response.status_code == 503
    assert json.loads(response.body) == {
        'error': 'No prefill or decode instances available',
    }
