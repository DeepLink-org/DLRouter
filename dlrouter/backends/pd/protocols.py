"""Protocols for shared PD executors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dlrouter.backends.base import PDRequestContext
    from dlrouter.core.node_manager import NodeManager


class PDExecutor(Protocol):
    """Common interface implemented by shared PD executors."""

    async def execute(
        self,
        request_data: dict[str, Any],
        endpoint: str,
        stream: bool,
        context: PDRequestContext,
    ) -> Any:
        """Execute a backend-owned PD request."""


class DualDispatchTransport(Protocol):
    """Transport required by dual-dispatch PD execution."""

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward a non-stream request to a backend node."""

    def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Forward a stream request to a backend node."""


class DualDispatchAdapter(Protocol):
    """Request adapter required by dual-dispatch PD execution."""

    def build_request(
        self,
        request_data: dict[str, Any],
        *,
        prefill_url: str,
        endpoint: str,
    ) -> dict[str, Any]:
        """Build a request payload shared by prefill and decode."""


class TwoStageTransferTransport(Protocol):
    """Transport required by two-stage transfer PD execution."""

    async def forward_with_request_id(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ) -> Any:
        """Forward a non-stream request with a shared request id."""

    def stream_forward_with_request_id(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        request_id: str,
    ) -> AsyncIterator[bytes]:
        """Forward a stream request with a shared request id."""


class TwoStageTransferAdapter(Protocol):
    """Request adapter required by two-stage transfer PD execution."""

    def build_request_id(
        self,
        prefill_url: str,
        decode_url: str,
        node_manager: NodeManager,
    ) -> str:
        """Build the connector-specific request id used across both stages."""

    def build_prefill_request(
        self,
        request_data: dict[str, Any],
        request_id: str,
        aborted_request_ids: list[str],
    ) -> dict[str, Any]:
        """Build the prefill-stage request payload."""

    def extract_transfer_context(
        self,
        prefill_response_json: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Extract optional transfer context from the prefill response."""

    def inject_decode_request(
        self,
        request_data: dict[str, Any],
        transfer_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the decode-stage request payload."""
