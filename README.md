# DLRouter

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()

DLRouter is an OpenAI-compatible inference gateway for large language model
backends. It routes requests across LMDeploy, vLLM, and SGLang instances with
pluggable routing strategies, runtime node management, health checks, and
Prefill/Decode disaggregation support.

Use DLRouter when you want one API endpoint in front of multiple LLM serving
nodes, while keeping backend-specific DistServe / PD orchestration out of your
application code.

## Highlights

- **OpenAI-compatible API**: `/v1/models`, `/v1/chat/completions`, and
  `/v1/completions`.
- **Multiple routing policies**: round-robin, weighted random, consistent hash,
  latency-aware routing, and prefix-cache-aware routing.
- **Multi-backend support**: LMDeploy, vLLM, and SGLang through a pluggable
  backend adapter interface.
- **DistServe / PD disaggregation**: backend-owned Prefill/Decode flows for
  LMDeploy, vLLM, and SGLang.
- **Dynamic node management**: register, remove, inspect, and terminate backend
  nodes through REST APIs.
- **Health checking and lazy model discovery**: unhealthy nodes are removed
  after consecutive failures, and model lists can be discovered after a backend
  becomes ready.
- **Optional authentication and TLS**: Bearer-token API keys and SSL/TLS support
  are available through CLI and environment configuration.

## Supported Backends

| Backend | Hybrid forwarding | DistServe / PD | Discovery modes | Notes |
|---|---:|---:|---|---|
| LMDeploy | Yes | Yes | External node registration | Uses LMDeploy PD connection pool and RDMA migration when available. |
| vLLM | Yes | Yes | Static, heartbeat | Supports two-stage KV transfer and static NIXL DP-aware rank routing. |
| SGLang | Yes | Yes | Static | Uses bootstrap dual dispatch with aligned prefill bootstrap ports. |
| NanoDeploy | Yes | Yes | dlslime-ctrl (`nanoctrl`) | Hybrid `nanodeploy serve` nodes; auto-discovery when `--ctrl_address` is set. |

DLRouter is configured with one backend type per router process through
`--backend`. Run multiple router processes if you need separate backend types at
the same time.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Python 3.9 or newer is required.

## Quick Start

This example starts DLRouter in vLLM hybrid mode, registers one vLLM server, and
sends an OpenAI-compatible chat request through DLRouter.

Start a vLLM server:

```bash
vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8100 \
  --served-model-name Qwen3-4B

# For single-node setups without Ray, add:
#   --distributed-executor-backend mp
```

Start DLRouter:

```bash
python -m dlrouter \
  --serving_strategy hybrid \
  --backend vllm
```

Register the backend node:

```bash
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://127.0.0.1:8100"}'
```

Send a request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### NanoDeploy with dlslime-ctrl discovery

Start the control plane and a NanoDeploy OpenAI server (see NanoDeploy
`nanodeploy serve`), then run DLRouter with auto-discovery:

```bash
dlslime-ctrl server --redis-url redis://127.0.0.1:6379

nanodeploy serve /path/to/model \
  --host 0.0.0.0 --port 8100 \
  --served-model-name Qwen3-4B \
  --ctrl-address 127.0.0.1:4479

pip install -e ".[nanodeploy]"   # pulls dlslime for NanoCtrlClient

python -m dlrouter \
  --backend nanodeploy \
  --serving_strategy hybrid \
  --ctrl_address 127.0.0.1:4479
```

DLRouter polls dlslime-ctrl for entities with kind `nanodeploy` and registers
their HTTP endpoints. Use the same `model` name as `--served-model-name` in
requests. Manual registration still works via `POST /nodes/add` when
`--ctrl_address` is omitted.

Send a request (the served model name, model path, and its basename are all
accepted as the `model` value):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

DLRouter also installs a `dlrouter` console script, so `dlrouter ...` is
equivalent to `python -m dlrouter ...` after installation.

## Common Usage

### Routing Strategies

```bash
python -m dlrouter \
  --backend vllm \
  --serving_strategy hybrid \
  --routing_strategy min_expected_latency
```

Available strategies:

| Strategy | Description |
|---|---|
| `round_robin` | Sequentially cycle through nodes serving the requested model. |
| `random` | Weighted random selection. Nodes reporting higher speed receive more traffic. |
| `consistent_hash` | Route requests with the same key to the same node for affinity or cache locality. |
| `min_expected_latency` | Select the node with the lowest estimated latency: `unfinished_requests / speed`. |
| `min_observed_latency` | Select the node with the lowest recent average latency. |
| `prefix_cache` | Route by KV cache prefix locality to improve cache hit rate. |

### vLLM DistServe: Static P/D Lists

Use static mode when prefill and decode HTTP endpoints are known at router
startup. DLRouter infers static discovery when both `--prefill_urls` and
`--decode_urls` are provided.

```bash
python -m dlrouter \
  --serving_strategy distserve \
  --backend vllm \
  --prefill_urls "http://prefill-1:30000,http://prefill-2:30000" \
  --decode_urls "http://decode-1:30000,http://decode-2:30000" \
  --models "Qwen3-32B" \
  --disable_cache_status
```

For NIXL intra-node data parallel routing, set
`--intra_node_data_parallel_size` to the local DP size. DLRouter expands each
physical URL into `url@rank` logical nodes for routing state, strips the suffix
before forwarding, and sends `X-data-parallel-rank` to vLLM.

```bash
python -m dlrouter \
  --serving_strategy distserve \
  --backend vllm \
  --prefill_urls "http://prefill-node:8001" \
  --decode_urls "http://decode-node:8002" \
  --models "Qwen3-32B" \
  --intra_node_data_parallel_size 8 \
  --disable_cache_status
```

If you change the DP size between router restarts, clear the persisted node
cache or use `--disable_cache_status` to avoid stale `url@rank` entries.

### vLLM DistServe: Heartbeat Discovery

When neither `--prefill_urls` nor `--decode_urls` is provided, vLLM DistServe
uses heartbeat discovery. P/D instances register themselves with DLRouter by
publishing HTTP and ZMQ addresses.

```bash
python -m dlrouter \
  --serving_strategy distserve \
  --backend vllm \
  --zmq_host 0.0.0.0 \
  --zmq_port 30001 \
  --models "Qwen3-32B" \
  --disable_cache_status
```

In heartbeat mode, a node enters the routable set only after DLRouter resolves
its model information. If a restarted node sends heartbeats before its HTTP API
is ready, registration is skipped temporarily and retried by later heartbeats.

### SGLang DistServe

SGLang currently uses static discovery in DLRouter. Provide both prefill and
decode URL lists. `--prefill_bootstrap_ports` is aligned with
`--prefill_urls`; if omitted, each prefill defaults to `8998`.

```bash
python -m dlrouter \
  --serving_strategy distserve \
  --backend sglang \
  --prefill_urls "http://prefill-1:13700,http://prefill-2:13700" \
  --decode_urls "http://decode-1:13701,http://decode-2:13701" \
  --prefill_bootstrap_ports "8998,8998" \
  --models "Qwen3-32B" \
  --disable_cache_status
```

DLRouter injects `bootstrap_host`, `bootstrap_port`, and `bootstrap_room` into
the request body, sends the decorated request to prefill and decode
concurrently, and returns the decode response.

### LMDeploy DistServe

LMDeploy DistServe relies on externally registered prefill and decode nodes.
DLRouter selects P/D nodes through `NodeManager`; no separate discovery component
is created by the app factory.

```bash
python -m dlrouter \
  --serving_strategy distserve \
  --backend lmdeploy \
  --migration_protocol RDMA \
  --link_type RoCE
```

LMDeploy-specific PD features require the LMDeploy disaggregation dependencies to
be installed in the runtime environment.

## CLI Reference

### Common Options

| Option | Default | Description |
|---|---|---|
| `--server_name` | `0.0.0.0` | Bind address. |
| `--server_port` | `8000` | Listen port. |
| `--backend` | `lmdeploy` | Backend type: `lmdeploy`, `vllm`, or `sglang`. |
| `--routing_strategy` | `min_expected_latency` | Request routing strategy. |
| `--serving_strategy` | `hybrid` | Serving mode: `hybrid` or `distserve`. |
| `--api_keys` | `None` | Comma-separated Bearer tokens for API authentication. |
| `--ssl` | `False` | Enable SSL. Requires `SSL_KEYFILE` and `SSL_CERTFILE`. |
| `--log_level` | `INFO` | DLRouter log level. |
| `--disable_cache_status` | `False` | Disable persisted node status. |
| `--config_path` | `None` | Custom node status persistence file. |
| `--workers` | `1` | Number of worker processes. Values greater than 1 use Gunicorn. |

### Backend Options

Backend-specific options are added dynamically and are visible with `--help`
after selecting a backend.

| Backend | Option | Default | Description |
|---|---|---|---|
| LMDeploy | `--migration_protocol` | `RDMA` | PD migration protocol. |
| LMDeploy | `--link_type` | `RoCE` | RDMA link type: `RoCE` or `IB`. |
| LMDeploy | `--with_gdr` | `True` | Enable GPU Direct RDMA. |
| LMDeploy | `--dummy_prefill` | `False` | Use dummy prefill for testing. |
| vLLM | `--zmq_host` | `0.0.0.0` | ZMQ discovery bind host. |
| vLLM | `--zmq_port` | `30001` | ZMQ discovery port. |
| vLLM | `--zmq_ping_timeout` | `5` | ZMQ instance ping timeout in seconds. |
| vLLM | `--prefill_urls` | `None` | Comma-separated prefill URLs for static mode. |
| vLLM | `--decode_urls` | `None` | Comma-separated decode URLs for static mode. |
| vLLM | `--models` | `None` | Comma-separated model names. |
| vLLM | `--intra_node_data_parallel_size` | `1` | Static NIXL DP-aware logical rank count per physical URL. |
| SGLang | `--prefill_urls` | `None` | Comma-separated SGLang prefill HTTP URLs. |
| SGLang | `--decode_urls` | `None` | Comma-separated SGLang decode HTTP URLs. |
| SGLang | `--prefill_bootstrap_ports` | `8998 per prefill` | Comma-separated bootstrap ports aligned with prefill URLs. |
| SGLang | `--models` | `None` | Comma-separated model names. |

## API Reference

### Inference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Router health check. |
| `GET` | `/v1/models` | List available models across registered nodes. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion endpoint. |
| `POST` | `/v1/completions` | OpenAI-compatible text completion endpoint. |

### Node Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/nodes/status` | Show registered nodes and routing state. |
| `POST` | `/nodes/add` | Register a backend node. |
| `POST` | `/nodes/remove` | Remove a backend node. |
| `POST` | `/nodes/terminate` | Terminate and remove a backend node. |
| `POST` | `/nodes/terminate_all` | Terminate all registered nodes. |

Node registration can provide only a URL, or a URL plus explicit `status`
metadata:

```bash
curl -X POST http://localhost:8000/nodes/add \
  -H "Content-Type: application/json" \
  -d '{"url": "http://backend-host:8000"}'
```

## Architecture

```text
Client (OpenAI SDK / curl)
        |
        v
   FastAPI routes
        |
        v
   ProxyEngine
        |
        +--> Hybrid: NodeManager -> RoutingStrategy -> Backend HTTP forward
        |
        +--> DistServe: Backend-owned PD executor
                 |
                 +--> LMDeploy PD / vLLM two-stage KV transfer / SGLang bootstrap
```

Key modules:

| Module | Responsibility |
|---|---|
| `dlrouter/api/` | FastAPI app, middleware, and OpenAI-compatible routes. |
| `dlrouter/core/proxy_engine.py` | Dispatches hybrid requests and delegates DistServe requests to backends. |
| `dlrouter/core/node_manager.py` | Maintains node state, model lists, request counters, and routing strategy instances. |
| `dlrouter/core/health_check.py` | Runs background health checks and lazy model discovery. |
| `dlrouter/routing/` | Pluggable routing strategy implementations. |
| `dlrouter/backends/` | Backend adapters, shared HTTP transport, and PD execution helpers. |

Backend adapters share the async HTTP transport layer in
`dlrouter/backends/http.py` for normal forwarding, streaming forwarding, health
checks, session lifecycle, and backend-specific stream framing. Backend-specific
logic remains in each backend package.

## Discovery Semantics

| Mode | Behavior |
|---|---|
| `HYBRID` | Backend instances are registered explicitly, usually through `/nodes/add`. |
| `DISTSERVE + vLLM + static` | Providing both `prefill_urls` and `decode_urls` selects static discovery. |
| `DISTSERVE + vLLM + heartbeat` | Providing neither URL list selects heartbeat discovery. |
| `DISTSERVE + SGLang` | Static P/D lists are required; heartbeat discovery is not used. |
| `DISTSERVE + LMDeploy` | P/D nodes are selected from `NodeManager`; no router-startup discovery object is created. |

Providing only one of `prefill_urls` or `decode_urls` is treated as a
configuration error.

## Environment Variables

| Variable | Description |
|---|---|
| `DLROUTER_HEARTBEAT_EXPIRATION` | Heartbeat timeout in seconds. Default: `90`. |
| `DLROUTER_HEALTH_CHECK_TIMEOUT` | Per-node health-check HTTP timeout in seconds. Default: `30`. |
| `DLROUTER_HEALTH_CHECK_MAX_FAILURES` | Consecutive failures before removing a node. Default: `3`. |
| `DLROUTER_AIOHTTP_TIMEOUT` | HTTP request timeout to backends in seconds. Default: `1800`. |
| `UVICORN_LOG_LEVEL` | Uvicorn log level. Default: `info`. |
| `SSL_KEYFILE` | SSL key file path when `--ssl` is enabled. |
| `SSL_CERTFILE` | SSL certificate file path when `--ssl` is enabled. |

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

# Type-check
make type-check

# Run tests
make test

# Run all checks
make all
```

The test suite lives under `tests/` and covers backend contracts, routing
strategies, service discovery, health checks, and PD executors.

## Current Limitations

- One DLRouter process is configured for one backend type at startup.
- SGLang DistServe currently uses static discovery only.
- LMDeploy PD features require LMDeploy disaggregation dependencies in the
  runtime environment.
- `fetch_models()` is synchronous in the current backend contract because node
  registration and lazy health-check discovery call it synchronously.

## Acknowledgements

DLRouter draws inspiration from these open-source projects:

- [LMDeploy](https://github.com/InternLM/lmdeploy), especially its proxy and PD
  disaggregation design.
- [vLLM](https://github.com/vllm-project/vllm), including router and cache-aware
  load-balancing ideas.
- [SGLang](https://github.com/sgl-project/sglang), especially router and mini
  load-balancer patterns for bootstrap-based PD proxying.

Thanks to the developers and contributors of these projects for their work in
the LLM inference ecosystem.

## License

Apache-2.0
