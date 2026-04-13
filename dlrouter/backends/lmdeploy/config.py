"""Configuration models for the LMDeploy backend."""

from pydantic import BaseModel


class LMDeployPDConfig(BaseModel):
    """LMDeploy PD disaggregation config."""

    migration_protocol: str = 'RDMA'
    link_type: str = 'RoCE'
    with_gdr: bool = True
    dummy_prefill: bool = False
