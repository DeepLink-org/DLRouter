"""Request-scoped state for PD executors."""

from dataclasses import dataclass, field


@dataclass
class TwoStageRequestState:
    """Request-scoped state for a two-stage prefill/decode flow."""

    request_id: str
    prefill_url: str
    decode_url: str
    aborted_request_ids: list[str] = field(default_factory=list)

    def mark_aborted(self) -> None:
        """Track that the current request should be treated as aborted."""
        self.aborted_request_ids.append(self.request_id)
