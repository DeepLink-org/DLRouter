"""Configuration models for the SGLang backend."""

from pydantic import BaseModel, Field

from dlrouter.constants import ServiceDiscoveryMode


class SGLangPDConfig(BaseModel):
    """SGLang PD disaggregation config."""

    discovery_mode: ServiceDiscoveryMode = ServiceDiscoveryMode.STATIC
    models: list[str] = Field(default_factory=list)
    prefill_urls: list[str] = Field(default_factory=list)
    decode_urls: list[str] = Field(default_factory=list)
    prefill_bootstrap_ports: list[int] = Field(default_factory=list)
