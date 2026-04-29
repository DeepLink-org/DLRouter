"""Tests for shared PD executor protocol imports."""

from dlrouter.backends.pd.protocols import (
    DualDispatchAdapter,
    DualDispatchTransport,
    PDExecutor,
    TwoStageTransferAdapter,
    TwoStageTransferTransport,
)


def test_pd_protocols_are_importable() -> None:
    assert DualDispatchAdapter is not None
    assert DualDispatchTransport is not None
    assert PDExecutor is not None
    assert TwoStageTransferAdapter is not None
    assert TwoStageTransferTransport is not None
