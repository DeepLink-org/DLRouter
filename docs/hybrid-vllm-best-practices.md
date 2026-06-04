# Hybrid + vLLM 最佳实践

DLRouter 统一 OpenAI 兼容入口，对接一台或多台 vLLM 实例。Hybrid 模式每台 vLLM 独立完成 Prefill + Decode，不做 P/D 分离（分离请看 DistServe）。

---

## 快速启动

**1. 起 vLLM**

```bash
vllm serve /data/models/Qwen3-4B \
  --host 0.0.0.0 \
  --port 8100 \
  --served-model-name Qwen3-4B
  # 单机无 Ray 加: --distributed-executor-backend mp
  # 多卡加: --tensor-parallel-size 4
```

**2. 起 DLRouter**

```bash
dlrouter \
  --backend vllm \
  --serving_strategy hybrid \
  --server_port 8000
```

> **`--disable_cache_status`**：加上后每次重启从空表开始，需要重新注册节点。
> 不加（默认）则节点状态持久化到 `router_config.json`，Router 重启后自动恢复已注册节点。
> 节点地址稳定时不加更方便；频繁扩缩容或想每次干净启动时再加。

**3. 注册节点**

```bash
curl -X POST http://127.0.0.1:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8100"}'
# 返回: {"message":"Added successfully."}
```

**4. 发请求**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128,
    "stream": false
  }'
```

> **关键**：`model` 字段必须与 vLLM 启动时的 `--served-model-name` 一致。

---

## 验证

```bash
curl http://127.0.0.1:8000/health          # 存活
curl http://127.0.0.1:8000/v1/models       # 聚合模型列表
curl http://127.0.0.1:8000/nodes/status    # 节点状态（role/models/unfinished/speed）
```

Router 日志中每次转发会打：`Dispatching to http://127.0.0.1:8100 (model=Qwen3-4B)`

---

## 多副本注册

多台 vLLM 同样 `--served-model-name`，依次 add 即可：

```bash
for url in \
  "http://10.0.0.11:8100" \
  "http://10.0.0.12:8100" \
  "http://10.0.0.13:8100"
do
  curl -s -X POST http://127.0.0.1:8000/nodes/add \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"${url}\"}" && echo
done
```

移除节点：

```bash
curl -X POST http://127.0.0.1:8000/nodes/remove \
  -H "Content-Type: application/json" \
  -d '{"url": "http://10.0.0.12:8100"}'
```

---

## 路由策略

`--routing_strategy` 启动时指定，默认 `min_expected_latency`：

| 策略 | 适合场景 |
|------|----------|
| `min_expected_latency`（默认） | 通用，选当前负载最低节点 |
| `round_robin` | 节点同质、负载均匀 |
| `consistent_hash` | 会话/用户亲和（同一 key 固定节点） |
| `prefix_cache` | RAG / 长 system prompt，提高 KV cache 命中 |
| `min_observed_latency` | 依赖历史 RTT 选节点 |
| `random` | 按 speed 加权随机 |

`consistent_hash` 时请求头 `x-session-id` / `x-user-id` 优先，否则 fallback 到 body 里的 `user` 字段。

---

## 用户案例

### UC-1 单机联调

最小集：1 台 vLLM + Router，直接按快速启动操作。对比验证时对同一 JSON 同时打 `:8000`（经 Router）和 `:8100`（直连 vLLM），行为应一致。

---

### UC-2 三副本负载均衡

```bash
dlrouter --backend vllm --serving_strategy hybrid \
  --routing_strategy min_expected_latency
```

注册 3 个节点，单台宕机后连续 3 次 `/health` 失败自动剔除（由 `DLROUTER_HEALTH_CHECK_MAX_FAILURES=3` 控制）。

---

### UC-3 API Key 鉴权

> 你还没试过这个，下面把完整步骤列清楚。

**原理**：DLRouter 做鉴权，通过后明文转发给 vLLM。**Router 不会把 Bearer token 透传给 vLLM 后端**，两侧鉴权独立。

**Step 1 — 带 `--api_keys` 启动 Router**

```bash
dlrouter \
  --backend vllm \
  --serving_strategy hybrid \
  --server_port 8000 \
  --api_keys "my-secret-key"
  # 多个 key 逗号分隔: --api_keys "key-a,key-b,key-c"
```

启动后未带 token 的请求直接返回 401：

```bash
curl http://127.0.0.1:8000/health
# 401 Unauthorized
```

**Step 2 — 所有请求都要带 token**

注册节点：

```bash
curl -X POST http://127.0.0.1:8000/nodes/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"url": "http://127.0.0.1:8100"}'
```

查状态：

```bash
curl http://127.0.0.1:8000/nodes/status \
  -H "Authorization: Bearer my-secret-key"
```

推理请求：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128,
    "stream": false
  }'
```

OpenAI Python SDK：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="my-secret-key",   # 对应 --api_keys 里的某个 key
)
resp = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

**Step 3（可选）— 同时保护 vLLM 本身**

Router 不转发 token，vLLM 侧若要鉴权需单独配置（如 vLLM `--api-key`）。内网部署通常只保护 Router 对外入口即可，vLLM 只对 Router 开放。

---

### UC-4 滚动升级副本

逐台操作：`/nodes/remove` → 重启 vLLM → 确认 `/health` 正常 → `/nodes/add` 重新注册。其余副本持续对外服务。

> 重启的是 vLLM 而非 Router 时，不需要重启 Router；节点状态默认持久化在 `router_config.json`，Router 重启后自动恢复。

---

## 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `model_not_found` | `model` 与 `--served-model-name` 不一致，或节点未注册 | 查 `/v1/models`、`/nodes/status` |
| 注册成功但 `models` 为空 | vLLM 未就绪，`fetch_models` 失败 | 等健康检查懒发现（默认 90s 一轮），或先确认 vLLM `/v1/models` 可访问 |
| 502 / `BACKEND_ERROR` | vLLM 崩溃、OOM、超时 | 查 vLLM 日志；可调大 `DLROUTER_AIOHTTP_TIMEOUT` |
| 节点被自动摘掉 | 连续 3 次 `/health` 失败 | 恢复 vLLM 后重新 add |
| 401 鉴权失败 | 配置了 `--api_keys` 但请求未带 token | 加 `-H "Authorization: Bearer <key>"` |

---

## 相关文档

- [README.md](../README.md)：CLI 全量参数、DistServe、环境变量
- [heterogeneous-ppu-maca-lmdeploy.md](./heterogeneous-ppu-maca-lmdeploy.md)：LMDeploy 异构 PD
