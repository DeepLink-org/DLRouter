# 混推部署：PPU + MACA

在 Prefill/Decode 分离（DistServe）场景下，使用 DLRouter 作为统一入口，将 Prefill 部署在 PPU、Decode 部署在 MACA。本文档覆盖容器准备、DLRouter 部署与联调步骤。

## 目录

- [以 PPU 和 MACA 为例](#以-ppu-和-maca-为例)
- [1 对于 MACA 机器](#1-对于-maca-机器)
- [2 对于 PPU 机器](#2-对于-ppu-机器)
- [3 对于 DLRouter](#3-对于-dlrouter)
- [4 开始启动](#4-开始启动)

## 以 PPU 和 MACA 为例

架构概要：

- **PPU 节点**：运行 Prefill（P）实例（`--device cuda`）
- **MACA 节点**：运行 Decode（D）实例（`--device maca`）
- **DLRouter**：作为 Proxy，统一对外提供 OpenAI 兼容 API，并编排 PD 流量

### 前置配置

下文命令中的路径与 IP 请按实际环境替换。建议在 shell 中先导出：

```bash
# 宿主机上的模型目录（docker -v 左侧）
export HOST_MODEL_DIR=/data/models/Qwen3-32B

# 容器内模型路径（docker -v 右侧，须与 lmdeploy 命令中一致）
export MODEL_PATH=/models/Qwen3-32B

# 对外注册的模型名（与 curl / OpenAI API 中的 model 字段一致）
export MODEL_NAME=pd_test

# 各组件所在节点 IP（示例为内网地址，请改为你的环境）
export PROXY_HOST=192.168.1.10    # DLRouter
export PREFILL_HOST=192.168.1.10   # P 实例（PPU）
export DECODE_HOST=192.168.1.11    # D 实例（MACA）
```

启动容器时，将模型目录挂载进容器，例如：

```bash
-v "${HOST_MODEL_DIR}:${MODEL_PATH}"
```

P 与 D 实例必须加载同一套模型权重。`${MODEL_PATH}` 是容器内的模型路径，需在对应 P/D 容器中可访问；不同节点的宿主机挂载路径可以不同，只要最终指向同一版本权重即可。

---

## 1 对于 MACA 机器

### 1.1 手动拉取镜像（验证权限 + 网络）

显卡型号：**MetaX C500**

```bash
docker pull crpi-z3i6df2ze96lh4ld.cn-hangzhou.personal.cr.aliyuncs.com/deeplink_infer/deeplink_infer:lt_maca-20260421
```

### 1.2 启动容器

```bash
docker run -itd \
   --privileged \
   --ipc host \
   --cap-add SYS_PTRACE \
   --device=/dev/mem \
   --device=/dev/dri \
   --device=/dev/mxcd \
   --device=/dev/infiniband \
   --group-add video \
   --network=host \
   --shm-size 400g \
   --ulimit memlock=-1 \
   --security-opt seccomp=unconfined \
   --security-opt apparmor=unconfined \
   -h "$(hostname)" \
   --name maca-decode \
   -v "${HOST_MODEL_DIR}:${MODEL_PATH}" \
   -v /var/run/docker.sock:/var/run/docker.sock \
   --entrypoint /bin/bash \
   "crpi-z3i6df2ze96lh4ld.cn-hangzhou.personal.cr.aliyuncs.com/deeplink_infer/deeplink_infer:lt_maca-20260421"
```

---

## 2 对于 PPU 机器

### 2.1 PPU 操作

#### 准备镜像

在 PPU 节点上执行。显卡型号：**PPU-ZW810E**

```bash
docker pull crpi-z3i6df2ze96lh4ld.cn-hangzhou.personal.cr.aliyuncs.com/deeplink_infer/deeplink_infer:v1.5.2-vllm0.8.5
```

### 2.2 启动 PPU 容器

```bash
docker run --privileged=true \
  --name ppu-prefill \
  --device=/dev/alixpu_ppu0 \
  --device=/dev/alixpu_ppu1 \
  --device=/dev/alixpu \
  --device=/dev/alixpu_ctl \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --init -td \
  --shm-size=500g \
  -v "${HOST_MODEL_DIR}:${MODEL_PATH}" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  crpi-z3i6df2ze96lh4ld.cn-hangzhou.personal.cr.aliyuncs.com/deeplink_infer/deeplink_infer:v1.5.2-vllm0.8.5
```

### 2.3 容器内部进行操作

#### 配置环境

如果没有 RDMA，请先执行：

```bash
apt update && apt install -y iproute2 rdma-core ibverbs-utils
```

查看本机 RDMA link（示例输出）：

```text
root@ppu-node:~# rdma link show
link mlx5_2/1 state ACTIVE physical_state LINK_UP netdev eth10
link mlx5_3/1 state ACTIVE physical_state LINK_UP netdev eth12
link mlx5_4/1 state ACTIVE physical_state LINK_UP netdev eth14
link mlx5_5/1 state ACTIVE physical_state LINK_UP netdev eth16
link mlx5_bond_0/1 state ACTIVE physical_state LINK_UP netdev ens5f0np0
```

> **注意**：`mlx5_bond_0` 对应后续命令中的 `export SLIME_VISIBLE_DEVICES=mlx5_bond_0`。

---

## 3 对于 DLRouter

在任意 PPU/MACA 环境上（或基于上述任一镜像新创建的容器内）启动 Proxy：

```bash
git clone https://github.com/DeepLink-org/DLRouter.git
pip3 install -e .
```

---

## 4 开始启动

### 4.1 启动 Proxy

在 `PROXY_HOST` 对应节点上执行（`--server_name` 填该节点 IP）：

```bash
dlrouter \
  --server_name "${PROXY_HOST}" \
  --server_port 8000 \
  --routing_strategy min_expected_latency \
  --serving_strategy distserve \
  --disable_cache_status
```

默认 backend 为 `lmdeploy`，与下文 LMDeploy PD 实例配套使用。若需显式指定：

```bash
dlrouter ... --backend lmdeploy
```

#### 可能遇到端口占用问题

- 查看容器内进程：`lsof -i :8000`，然后 `kill -9 <PID>`
- 或更换端口：`8001` / …

### 4.2 启动 PPU 上的 P 实例

在 PPU 容器/节点内执行：

```bash
# 启用 RDMA 网络
export SLIME_VISIBLE_DEVICES=mlx5_bond_0

lmdeploy serve api_server \
  "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --server-name "${PREFILL_HOST}" \
  --server-port 23333 \
  --role Prefill \
  --proxy-url "http://${PROXY_HOST}:8000" \
  --backend pytorch \
  --device cuda \
  --cache-block-seq-len 16 \
  --tp 4
```

`--tp` 按模型与卡数调整；上例为 Qwen3-32B、TP=4 时的参考值。

### 4.3 启动 MACA 上的 D 实例

在 MACA 容器/节点内执行：

```bash
export SLIME_VISIBLE_DEVICES=mlx5_bond_0

lmdeploy serve api_server \
  "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --server-name "${DECODE_HOST}" \
  --server-port 23333 \
  --role Decode \
  --proxy-url "http://${PROXY_HOST}:8000" \
  --backend pytorch \
  --device maca \
  --cache-block-seq-len 16 \
  --tp 4
```

### 4.4 测试是否正确

```bash
curl -X POST "http://${PROXY_HOST}:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\", \"temperature\": 0, \"prompt\": \"Shanghai is a city that \", \"max_tokens\": 128, \"stream\": false}"
```

---

## 部署拓扑参考

| 组件 | 地址（示例变量） | 端口 | 角色 |
|------|------------------|------|------|
| DLRouter Proxy | `${PROXY_HOST}` | 8000 | 统一入口 |
| LMDeploy Prefill | `${PREFILL_HOST}` | 23333 | P（PPU / cuda） |
| LMDeploy Decode | `${DECODE_HOST}` | 23333 | D（MACA / maca） |
| 模型权重 | `${MODEL_PATH}`（容器内） | — | P/D 共用同一路径 |
