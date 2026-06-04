# DLRouter Documentation

Public-facing documentation for DLRouter users and contributors.

## Serving modes（部署模式）

| Guide | Mode | Backend | 说明 |
|-------|------|---------|------|
| [hybrid-vllm-best-practices.md](./hybrid-vllm-best-practices.md) | **Hybrid** | vLLM | 单入口、多 vLLM 副本、步骤与用户案例 |
| [heterogeneous-ppu-maca-lmdeploy.md](./heterogeneous-ppu-maca-lmdeploy.md) | **DistServe** | LMDeploy | PPU (Prefill) + MACA (Decode) 异构混推 |

## See also

- 项目根目录 [README.md](../README.md)：功能概览、CLI 与通用 DistServe 配置
