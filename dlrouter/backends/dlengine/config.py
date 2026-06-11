"""DLEngine backend configuration."""

from dataclasses import dataclass


@dataclass
class DLEngineConfig:
    """Configuration for DLEngine hybrid nodes via dlslime-ctrl."""

    ctrl_address: str | None = None
    ctrl_scope: str | None = None
    ctrl_kind: str = 'dlengine'
    discovery_poll_interval: float = 5.0
