"""Configuration models for DLRouter."""

from typing import Optional

from pydantic import BaseModel, Field

from dlrouter.constants import (
    BackendType,
    RoutingStrategy,
    ServingStrategy,
)


class BackendConfig(BaseModel):
    """Configuration for an inference backend."""

    type: BackendType = BackendType.LMDEPLOY
    extra: dict = Field(default_factory=dict)


class LMDeployPDConfig(BaseModel):
    """LMDeploy PD disaggregation config."""

    migration_protocol: str = 'RDMA'
    link_type: str = 'RoCE'
    with_gdr: bool = True
    dummy_prefill: bool = False


class VLLMPDConfig(BaseModel):
    """vLLM PD disaggregation config.

    This config is used when serving_strategy is VLLM_PD.
    vLLM PD mode uses ZMQ for service discovery and request_id
    encoding for KV cache transfer coordination.
    """

    zmq_port: int = 30001
    zmq_host: str = '0.0.0.0'
    ping_timeout_seconds: int = 5


class SSLConfig(BaseModel):
    """SSL configuration."""

    enabled: bool = False
    keyfile: Optional[str] = None
    certfile: Optional[str] = None


class RouterConfig(BaseModel):
    """Top-level router configuration."""

    server_name: str = '0.0.0.0'
    server_port: int = 8000
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    serving_strategy: ServingStrategy = ServingStrategy.HYBRID
    backend: BackendConfig = Field(default_factory=BackendConfig)
    pd_config: LMDeployPDConfig = Field(default_factory=LMDeployPDConfig)
    vllm_pd_config: VLLMPDConfig = Field(default_factory=VLLMPDConfig)
    ssl: SSLConfig = Field(default_factory=SSLConfig)
    api_keys: Optional[list[str]] = None
    log_level: str = 'INFO'
    cache_status: bool = True
    config_path: Optional[str] = None
