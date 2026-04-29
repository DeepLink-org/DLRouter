"""Tests for the public PD package API."""

from importlib.util import find_spec

from dlrouter.backends.pd import (
    DualDispatchExecutor,
    PDExecutor,
    PDPair,
    PDPairSelector,
    TwoStageRequestState,
    TwoStageTransferExecutor,
    no_pd_pair_response,
)
from dlrouter.backends.pd.selection import no_pd_pair_response as selection_no_pd_pair_response
from dlrouter.backends.pd.state import TwoStageRequestState as StateTwoStageRequestState


def test_pd_package_exports_common_runtime_api() -> None:
    assert DualDispatchExecutor is not None
    assert PDExecutor is not None
    assert PDPair is not None
    assert PDPairSelector is not None
    assert TwoStageRequestState is not None
    assert TwoStageTransferExecutor is not None
    assert no_pd_pair_response is not None


def test_pd_state_and_selection_modules_export_precise_helpers() -> None:
    assert StateTwoStageRequestState is TwoStageRequestState
    assert selection_no_pd_pair_response is no_pd_pair_response


def test_pd_support_catch_all_module_is_removed() -> None:
    assert find_spec('dlrouter.backends.pd.support') is None
