"""Configuration models for the vLLM backend."""

from typing import Literal

from pydantic import BaseModel, Field

from dlrouter.constants import ServiceDiscoveryMode


class VLLMPDConfig(BaseModel):
    """vLLM PD disaggregation config."""

    discovery_mode: ServiceDiscoveryMode = ServiceDiscoveryMode.HEARTBEAT
    pd_protocol: Literal['two_stage_kv_transfer'] = 'two_stage_kv_transfer'
    zmq_host: str = '0.0.0.0'
    zmq_port: int = 30001
    ping_timeout_seconds: int = 5
    models: list[str] = Field(default_factory=list)
    prefill_urls: list[str] = Field(default_factory=list)
    decode_urls: list[str] = Field(default_factory=list)
