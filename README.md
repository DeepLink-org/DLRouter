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
- **Multi-Backend Architecture** — Pluggable backend adapters via the `BaseBackend` interface. Currently supported:
  - **LMDeploy** (including PD disaggregation / DistServe)
  - **vLLM** (standard OpenAI-compatible API forwarding)
- **PD Disaggregation (DistServe)** — First-class support for LMDeploy's Prefill-Decode separation, with automatic PD connection management and migration request handling.
- **Dynamic Node Management** — Register, remove, and terminate backend nodes at runtime via REST API.
- **Automatic Health Checks** — Background heartbeat thread removes unhealthy nodes automatically.
- **API Key Authentication** — Optional Bearer token authentication for all endpoints.
- **SSL / TLS Support** — Enable HTTPS via environment variables.

## Project Structure

```
DLRouter/
├── dlrouter/
│   ├── __main__.py            # CLI entry point
│   ├── config.py              # Configuration models (Pydantic)
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
│   │   ├── lmdeploy_backend.py # LMDeploy adapter (+ PD disagg)
│   │   ├── vllm_backend.py    # vLLM adapter
│   │   └── factory.py         # Backend factory
│   ├── core/
│   │   ├── node_manager.py    # Node registry & lifecycle
│   │   ├── proxy_engine.py    # Request dispatch (Hybrid / DistServe)
│   │   └── health_check.py    # Background health checker
│   ├── models/
│   │   ├── node.py            # Node / NodeStatus models
│   │   └── protocol.py        # OpenAI-compatible request/response models
│   └── routing/
│       ├── base.py            # Abstract routing strategy
│       ├── round_robin.py
│       ├── random_strategy.py
│       ├── consistent_hash.py
│       ├── load_aware.py      # min_expected / min_observed latency
│       └── factory.py         # Strategy factory
├── tests/
│   ├── backends/
│   │   └── test_vllm_backend.py   # vLLM backend unit tests
│   ├── routing/
│   │   └── test_routing.py        # Routing strategy unit tests
│   └── utils/
│       └── test_request_key.py    # Request key extraction tests
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
| `--migration_protocol` | `RDMA` | PD migration protocol |
| `--link_type` | `RoCE` | RDMA link type (`RoCE` / `IB`) |
| `--dummy_prefill` | `False` | Use dummy prefill (for testing) |

### Examples

```bash
# Round-robin routing on port 9000
python -m dlrouter --server_port 9000 --routing_strategy round_robin

# Consistent hash routing with API key
python -m dlrouter --routing_strategy consistent_hash --api_keys "sk-abc123,sk-def456"

# PD disaggregation mode (DistServe)
python -m dlrouter --serving_strategy distserve --link_type RoCE
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
| `GET` | `/nodes/terminate_all` | Terminate all nodes |

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

## License

Apache-2.0
