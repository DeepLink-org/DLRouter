# DLRouter

A high-performance router / load balancer for large language model (LLM) inference backends, providing an OpenAI-compatible API gateway with pluggable routing strategies and multi-backend support.

## Features

- **OpenAI-Compatible API** — Exposes `/v1/models`, `/v1/chat/completions`, `/v1/completions` and health endpoints, making it a drop-in proxy for any OpenAI SDK client.
- **Multiple Routing Strategies**
  - `round_robin` — Cycle through available nodes sequentially.
  - `random` — Weighted random selection based on node speed.
  - `consistent_hash` — Route requests with the same key (e.g. user id) to the same node.
  - `min_expected_latency` — Pick the node with the lowest estimated latency (`unfinished / speed`).
  - `min_observed_latency` — Pick the node with the lowest measured average latency.
  - `prefix_cache` — Prefix-aware routing that routes requests with shared prefixes to the same node to maximize KV cache reuse.
- **Multi-Backend Architecture** — Pluggable backend adapters via the `BaseBackend` interface. Currently supported:
  - **LMDeploy** (including PD disaggregation / DistServe)
  - **vLLM** (hybrid OpenAI-compatible forwarding via explicitly registered nodes, plus DistServe two-stage PD orchestration with static or heartbeat discovery)
- **PD Disaggregation (DistServe)** — First-class support for LMDeploy's and vLLM's Prefill-Decode separation, with automatic PD connection management and migration request handling.
- **Backend-Owned DistServe Flow** — `ProxyEngine` only dispatches DistServe requests; each backend owns its own PD orchestration (`LMDeploy` via `NodeManager`, `vLLM` via service discovery).
- **Dynamic Node Management** — Register, remove, and terminate backend nodes at runtime via REST API.
- **Automatic Health Checks** — Background heartbeat thread removes unhealthy nodes automatically.
- **API Key Authentication** — Optional Bearer token authentication for all endpoints.
- **SSL / TLS Support** — Enable HTTPS via environment variables.
- **Clear Discovery Semantics** — `HYBRID` nodes are added explicitly; `DISTSERVE` uses either `static` or `heartbeat` discovery for vLLM PD. In vLLM heartbeat mode, a node becomes routable only after model information is available.

## Project Structure

```
DLRouter/
├── dlrouter/
│   ├── __main__.py            # CLI entry point
│   ├── config.py              # Configuration models (RouterConfig, SSLConfig)
│   ├── constants.py           # Enums & constants
│   ├── logger.py              # Logging utilities
│   ├── api/
│   │   ├── app.py             # FastAPI application factory
│   │   ├── middleware.py       # API key authentication
│   │   └── routes/
│   │       ├── models.py      # GET  /health, /v1/models
│   │       ├── chat.py        # POST /v1/chat/completions
│   │       ├── completions.py # POST /v1/completions
│   │       └── nodes.py       # Node management endpoints
│   ├── backends/
│   │   ├── base.py            # Abstract backend interface
│   │   ├── lmdeploy/          # LMDeploy backend package (+ PD disagg)
│   │   ├── vllm/              # vLLM backend package
│   │   └── factory.py         # Backend factory
│   ├── core/
│   │   ├── node_manager.py    # Node registry & lifecycle
│   │   ├── proxy_engine.py    # Request dispatch (Hybrid / DistServe)
│   │   ├── health_check.py    # Background health checker
│   │   └── service_discovery/ # Static + heartbeat discovery abstractions for PD
│   ├── models/
│   │   ├── node.py            # Node / NodeStatus models
│   │   └── protocol.py        # OpenAI-compatible request/response models
│   └── routing/
│       ├── base.py            # Abstract routing strategy
│       ├── round_robin.py
│       ├── random_strategy.py
│       ├── consistent_hash.py
│       ├── load_aware.py      # min_expected / min_observed latency
│       ├── prefix_cache.py    # Prefix cache aware routing
│       └── factory.py         # Strategy factory
├── tests/
│   ├── backends/
│   │   ├── test_backend_contracts.py   # Backend interface contract tests
│   │   ├── test_backend_definitions.py # Backend definition tests
│   │   ├── test_lmdeploy_backend.py    # LMDeploy backend PD tests
│   │   ├── test_vllm_backend.py        # vLLM backend unit tests
│   │   ├── test_vllm_kv_transfer.py    # KV transfer adapter tests
│   │   ├── test_vllm_pair_selection.py # PD pair selection tests
│   │   ├── test_vllm_request_id.py     # Request ID encoding tests
│   │   └── test_vllm_two_stage.py      # Two-stage PD executor tests
│   ├── core/
│   │   ├── test_health_check.py             # Health checker tests
│   │   ├── test_proxy_engine.py             # ProxyEngine delegation tests
│   │   ├── test_service_discovery_factory.py # Discovery factory tests
│   │   ├── test_service_discovery_registry.py # Unified registry tests
│   │   ├── test_static_discovery.py         # Static discovery tests
│   │   └── test_zmq_discovery.py            # ZMQ heartbeat discovery tests
│   ├── routing/
│   │   └── test_routing.py        # Routing strategy unit tests
│   ├── utils/
│   │   └── test_request_key.py    # Request key extraction tests
│   ├── test_app_vllm_discovery.py # App factory discovery inference tests
│   └── test_cli_backend_loading.py # CLI backend loading tests
├── Makefile                   # Dev commands (format, lint, test, etc.)
└── pyproject.toml             # Project metadata & tool configuration
```

## Quick Start

### Installation

```bash
# From source (editable mode)
pip install -e .

# With dev dependencies (ruff, pytest, mypy, pre-commit)
pip install -e ".[dev]"
```

### Launch the Router

```bash
# Default: listen on 0.0.0.0:8000, lmdeploy backend, min_expected_latency routing
python -m dlrouter

# Or use the installed CLI
dlrouter
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--server_name` | `0.0.0.0` | Bind address |
| `--server_port` | `8000` | Listen port |
| `--backend` | `lmdeploy` | Backend type (`lmdeploy` / `vllm`) |
| `--routing_strategy` | `min_expected_latency` | Routing strategy (see below) |
| `--serving_strategy` | `hybrid` | Serving mode (`hybrid` / `distserve`) |
| `--api_keys` | `None` | Comma-separated API keys for auth |
| `--ssl` | `False` | Enable SSL (requires `SSL_KEYFILE` & `SSL_CERTFILE` env vars) |
| `--log_level` | `INFO` | Logging level |
| `--disable_cache_status` | `False` | Disable node status persistence |
| `--config_path` | `None` | Custom path for config persistence file |
| `--workers` | `1` | Number of worker processes |

**Backend-specific options** (shown in `--help` when using that backend):

*LMDeploy options:*
| `--migration_protocol` | `RDMA` | PD migration protocol |
| `--link_type` | `RoCE` | RDMA link type (`RoCE` / `IB`) |
| `--with_gdr` | `True` | Enable GPU Direct RDMA |
| `--dummy_prefill` | `False` | Use dummy prefill (for testing) |

*vLLM options:*
| `--zmq_host` | `0.0.0.0` | ZMQ service discovery bind host |
| `--zmq_port` | `30001` | ZMQ service discovery port |
| `--zmq_ping_timeout` | `5` | ZMQ instance ping timeout (seconds) |
| `--prefill_urls` | `None` | Comma-separated prefill URLs (when set together with `--decode_urls`, DLRouter infers static mode) |
| `--decode_urls` | `None` | Comma-separated decode URLs (when set together with `--prefill_urls`, DLRouter infers static mode) |
| `--models` | `None` | Comma-separated model names (optional, auto-fetched from nodes) |

### Examples

```bash
# Round-robin routing on port 9000
python -m dlrouter --server_port 9000 --routing_strategy round_robin

# Consistent hash routing with API key
python -m dlrouter --routing_strategy consistent_hash --api_keys "sk-abc123,sk-def456"

# LMDeploy PD disaggregation mode (DistServe)
python -m dlrouter --serving_strategy distserve --backend lmdeploy --link_type RoCE

# vLLM PD disaggregation mode (heartbeat registration)
python -m dlrouter --serving_strategy distserve --backend vllm --zmq_port 30001

# vLLM PD disaggregation mode (static P/D lists)
python -m dlrouter --serving_strategy distserve --backend vllm \
  --prefill_urls "http://10.21.9.10:30000" \
  --decode_urls "http://10.21.9.15:30000"

# Use vLLM as backend in hybrid mode (register nodes via /nodes/add)
python -m dlrouter --backend vllm

# Multi workers
python -m dlrouter --workers 4
```

## vLLM Usage

### vLLM Hybrid

Use `hybrid` when you want DLRouter to forward requests to standard vLLM OpenAI-compatible instances.

```bash
# Start a single vLLM instance
vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8100 \
  --served-model-name Qwen3-4B

# Start DLRouter
python -m dlrouter \
  --serving_strategy hybrid \
  --backend vllm \
  --disable_cache_status

# Register the vLLM node
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8100"}'

# Send a request through DLRouter
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

`/nodes/add` can also include an explicit `status.models` payload, but for vLLM it is usually enough to register the node URL and let DLRouter fetch `/v1/models`.

### vLLM DistServe

Use `distserve` when prefill and decode are separated.

- `static` mode is for explicitly configured prefill/decode URLs.
- `heartbeat` mode is for self-registering prefill/decode instances that publish HTTP and ZMQ addresses.
- DLRouter infers `static` when both `--prefill_urls` and `--decode_urls` are provided.
- DLRouter infers `heartbeat` when neither URL list is provided.
- Providing only one of the two URL lists is treated as a configuration error.
- In vLLM heartbeat mode, DLRouter fetches model information before admitting a node into the routable set.
- If a restarted node is sending heartbeats before its HTTP API is ready, DLRouter temporarily skips registration and later heartbeats retry automatically.

Typical heartbeat startup:

```bash
python -m dlrouter \
  --serving_strategy distserve \
  --backend vllm \
  --zmq_host 0.0.0.0 \
  --zmq_port 30001 \
  --disable_cache_status
```

## API Reference

### Inference Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models across all nodes |
| `POST` | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| `POST` | `/v1/completions` | Text completion (OpenAI-compatible) |

### Node Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/nodes/status` | Show all nodes and their status |
| `POST` | `/nodes/add` | Register a new backend node |
| `POST` | `/nodes/remove` | Remove a registered node |
| `POST` | `/nodes/terminate` | Terminate and remove a node |
| `POST` | `/nodes/terminate_all` | Terminate all nodes |

### Discovery Semantics

- `HYBRID`: backend instances are registered explicitly, typically via `/nodes/add` or direct `NodeManager.add(...)`.
- `DISTSERVE + vLLM`: providing both `prefill_urls` and `decode_urls` selects `static`; providing neither selects `heartbeat`.
- `DISTSERVE + vLLM + heartbeat`: a node enters the routable set only after DLRouter has resolved its model information.
- `DISTSERVE + LMDeploy`: Prefill/Decode nodes are still selected from `NodeManager`; no separate discovery component is created.

### Usage Example

```bash
# Register a backend node
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://gpu-server-1:23333"}'

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "internlm2-chat-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Routing Strategies

| Strategy | Description |
|---|---|
| `round_robin` | Sequentially cycle through nodes serving the requested model. |
| `random` | Weighted random selection — nodes reporting higher speed receive more traffic. |
| `consistent_hash` | Hash-based routing that maps a request key (e.g. `user` field) to a fixed node. Useful for session affinity or cache locality. |
| `min_expected_latency` | Select the node with the lowest estimated latency: `unfinished_requests / speed`. |
| `min_observed_latency` | Select the node with the lowest average latency measured from recent requests. |
| `prefix_cache` | Routes requests with shared prompt prefixes to the same backend node to maximize KV cache utilization. Uses a Trie data structure for efficient prefix matching with load balancing fallback. |

## Environment Variables

| Variable | Description |
|---|---|
| `DLROUTER_HEARTBEAT_EXPIRATION` | Heartbeat interval in seconds (default: `90`) |
| `DLROUTER_AIOHTTP_TIMEOUT` | HTTP request timeout to backends in seconds (default: `1800`) |
| `UVICORN_LOG_LEVEL` | Uvicorn log level (default: `info`) |
| `SSL_KEYFILE` | Path to SSL key file (when `--ssl` is enabled) |
| `SSL_CERTFILE` | Path to SSL certificate file (when `--ssl` is enabled) |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
make format

# Lint
make lint

# Auto-fix lint issues
make fix

# Run tests
make test

# Run all checks (lint + format check + type check + test)
make all

# Install pre-commit hooks
make pre-commit-install
```

## Architecture

```
Client (OpenAI SDK / curl)
        │
        ▼
   ┌─────────┐
   │ DLRouter │──── API Layer (FastAPI)
   └────┬────┘
        │
   Routing Strategy
   (RR / Random / Hash / Load-aware)
        │
        ├──► Backend Node 1
        ├──► Backend Node 2
        └──► Backend Node 3
             (all nodes use the same backend type, configured via --backend)
```

**DistServe (PD Disaggregation) mode:**

`ProxyEngine` does not implement backend-specific PD logic anymore. It delegates every DistServe request to `backend.handle_pd_request(...)`, passing a small context object with `NodeManager`, optional `service_discovery`, and `request_key`.

*LMDeploy PD:*
```
Client
  │
  ▼
DLRouter
  │
  ├─ 1. Prefill ──► P Node (Prefill engine)
  │                    │
  │              KV Cache Migration (RDMA)
  │                    │
  └─ 2. Decode  ──► D Node (Decode engine) ──► Response
```

LMDeploy does not use a separate discovery component in DistServe mode. Prefill and decode nodes are selected from `NodeManager`, and the LMDeploy backend handles PD connection setup, migration request construction, and cleanup.

*vLLM PD (ZMQ Service Discovery):*
```
vLLM P/D Instances ──► ZMQ Register ──► DLRouter
                                           │
Client ─────────────────────────────────► DLRouter
  │                                          │
  ▼                                          ▼
Request ────► Prefill (max_tokens=1) ──► P Node
                  │
            request_id encoding (P_zmq → D_zmq)
                  │
              Decode ──────────────────► D Node ──► Response
```

vLLM DistServe keeps using backend-owned PD orchestration as well, but its node selection source is `service_discovery` rather than `NodeManager`. In heartbeat mode, a node is admitted only after model information has been resolved, so restarted nodes may briefly stay out of the routable set until their HTTP API is ready.

## Acknowledgements

This project draws inspiration from the following open-source projects:

- **[LMDeploy](https://github.com/InternLM/lmdeploy)** — The proxy implementation in `lmdeploy/serve/proxy/proxy.py` provided valuable reference for the routing architecture and PD disaggregation support.
- **[vLLM](https://github.com/vllm-project/vllm)** — The implementation of load balancing policies such as cache_aware in VLLM routers provides us with many references.

We extend our sincere thanks to the developers and contributors of these projects for their excellent work in the LLM inference ecosystem.

## License

Apache-2.0
