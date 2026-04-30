"""Request-scoped state for PD executors."""

from dataclasses import dataclass, field


@dataclass
class TwoStageRequestState:
    """Request-scoped state for a two-stage prefill/decode flow.

    aborted_request_ids is designed for retry-attempt bookkeeping
    within one logical request. Current executors do not perform retries;
    each execute() call starts with fresh request-local abort tracking.
    """

    request_id: str
    prefill_url: str
    decode_url: str
    aborted_request_ids: list[str] = field(default_factory=list)

    def mark_aborted(self) -> None:
        """Append this request id to the request-local abort list."""
        self.aborted_request_ids.append(self.request_id)
