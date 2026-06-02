"""NanoDeploy backend configuration."""

from dataclasses import dataclass


@dataclass
class NanoDeployConfig:
    """Configuration for NanoDeploy hybrid nodes via dlslime-ctrl."""

    ctrl_address: str | None = None
    ctrl_scope: str | None = None
    ctrl_kind: str = 'nanodeploy'
    discovery_poll_interval: float = 5.0
