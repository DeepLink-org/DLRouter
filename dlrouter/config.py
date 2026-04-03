"""Configuration models for DLRouter."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from dlrouter.constants import (
    BackendType,
    RoutingStrategy,
    ServingStrategy,
)


class SSLConfig(BaseModel):
    """SSL configuration."""

    enabled: bool = False
    keyfile: Optional[str] = None
    certfile: Optional[str] = None


class RouterConfig(BaseModel):
    """Top-level router configuration.

    Backend-specific configurations are stored in backend_config
    and parsed by the corresponding backend class.
    """

    server_name: str = '0.0.0.0'
    server_port: int = 8000
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    serving_strategy: ServingStrategy = ServingStrategy.HYBRID
    backend_type: BackendType = BackendType.LMDEPLOY
    backend_config: dict[str, Any] = Field(default_factory=dict)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    api_keys: Optional[list[str]] = None
    log_level: str = 'INFO'
    cache_status: bool = True
    config_path: Optional[str] = None
