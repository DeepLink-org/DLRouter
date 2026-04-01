"""DLRouter CLI entry point.

Usage:
    python -m dlrouter [OPTIONS]
    dlrouter [OPTIONS]

Examples:
    # Start with defaults (single process)
    python -m dlrouter

    # Multi-process mode with 4 workers
    python -m dlrouter --workers 4

    # Custom port and strategy
    python -m dlrouter --server_port 9000 \
        --routing_strategy round_robin

    # With PD disaggregation
    python -m dlrouter --serving_strategy distserve
"""

import os
import sys
from typing import Literal, Optional, Union

import fire
import uvicorn

from dlrouter.api.app import create_app
from dlrouter.config import (
    BackendConfig,
    LMDeployPDConfig,
    RouterConfig,
    SSLConfig,
)
from dlrouter.constants import (
    BackendType,
    RoutingStrategy,
    ServingStrategy,
)
from dlrouter.logger import get_logger


logger = get_logger('dlrouter')


def serve(
    server_name: str = '0.0.0.0',
    server_port: int = 8000,
    backend: Literal['lmdeploy', 'vllm'] = 'lmdeploy',
    routing_strategy: Literal[
        'round_robin',
        'random',
        'consistent_hash',
        'min_expected_latency',
        'min_observed_latency',
    ] = 'min_expected_latency',
    serving_strategy: Literal['hybrid', 'distserve'] = 'hybrid',
    api_keys: Optional[Union[list[str], str]] = None,
    ssl: bool = False,
    log_level: str = 'INFO',
    disable_cache_status: bool = False,
    config_path: Optional[str] = None,
    migration_protocol: str = 'RDMA',
    link_type: Literal['RoCE', 'IB'] = 'RoCE',
    with_gdr: bool = True,
    dummy_prefill: bool = False,
    workers: int = 1,
):
    """Launch the DLRouter proxy server.

    Args:
        server_name: Bind address. Default 0.0.0.0.
        server_port: Listen port. Default 8000.
        backend: Inference backend type.
        routing_strategy: Request routing strategy.
        serving_strategy: Serving mode.
        api_keys: Optional API keys (comma-separated
            string or list).
        ssl: Enable SSL (requires SSL_KEYFILE and
            SSL_CERTFILE env vars).
        log_level: Logging level.
        disable_cache_status: Disable config caching.
        config_path: Path to config persistence file.
        migration_protocol: PD migration protocol.
        link_type: RDMA link type.
        with_gdr: Enable GPU Direct RDMA.
        dummy_prefill: Use dummy prefill for testing.
        workers: Number of worker processes. Use >1 for
            multi-process mode (requires gunicorn).
    """
    # Parse api_keys
    if isinstance(api_keys, str):
        api_keys = api_keys.split(',')

    # Build config
    config = RouterConfig(
        server_name=server_name,
        server_port=server_port,
        routing_strategy=RoutingStrategy(routing_strategy),
        serving_strategy=ServingStrategy(serving_strategy),
        backend=BackendConfig(type=BackendType(backend)),
        pd_config=LMDeployPDConfig(
            migration_protocol=migration_protocol,
            link_type=link_type,
            with_gdr=with_gdr,
            dummy_prefill=dummy_prefill,
        ),
        ssl=SSLConfig(enabled=ssl),
        api_keys=api_keys,
        log_level=log_level,
        cache_status=not disable_cache_status,
        config_path=config_path,
    )

    # Set log level
    logger.setLevel(log_level.upper())

    # Multi-process mode with Gunicorn
    if workers > 1:
        _run_with_gunicorn(
            server_name=server_name,
            server_port=server_port,
            workers=workers,
            log_level=log_level,
            ssl=ssl,
            config=config,
        )
        return

    # Single-process mode with Uvicorn
    # Create app
    app = create_app(config)

    # SSL
    ssl_keyfile = None
    ssl_certfile = None
    if ssl:
        ssl_keyfile = os.environ.get('SSL_KEYFILE')
        ssl_certfile = os.environ.get('SSL_CERTFILE')

    # Launch
    uv_log = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(
        app=app,
        host=server_name,
        port=server_port,
        log_level=uv_log,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


def _run_with_gunicorn(
    server_name: str,
    server_port: int,
    workers: int,
    log_level: str,
    ssl: bool,
    config: RouterConfig,
) -> None:
    """Run DLRouter with Gunicorn for multi-process mode.

    Gunicorn provides:
    - Multiple worker processes for CPU parallelism
    - Automatic worker management and restart
    - Better resource utilization
    """
    try:
        import gunicorn.app.base
    except ImportError:
        print(
            'Error: gunicorn is required for multi-process mode.\nInstall it with: pip install gunicorn',
            file=sys.stderr,
        )
        sys.exit(1)

    class DLRouterApplication(gunicorn.app.base.BaseApplication):
        """Custom Gunicorn application."""

        def __init__(self, app_factory, options=None):
            self.app_factory = app_factory
            self.options = options or {}
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return self.app_factory()

    # Build Gunicorn options
    bind = f'{server_name}:{server_port}'
    gunicorn_options = {
        'bind': bind,
        'workers': workers,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 120,
        'keepalive': 5,
        'loglevel': log_level.lower(),
        'accesslog': '-',
        'errorlog': '-',
        'proc_name': 'dlrouter',
        'max_requests': 10000,
        'max_requests_jitter': 1000,
    }

    # SSL configuration
    if ssl:
        ssl_keyfile = os.environ.get('SSL_KEYFILE')
        ssl_certfile = os.environ.get('SSL_CERTFILE')
        if ssl_keyfile:
            gunicorn_options['keyfile'] = ssl_keyfile
        if ssl_certfile:
            gunicorn_options['certfile'] = ssl_certfile

    print(
        f'Starting DLRouter with {workers} workers on {bind}...',
    )

    # Store config in environment for worker processes
    os.environ['DLROUTER_CONFIG_JSON'] = config.model_dump_json()

    # Create app factory that reads config from environment
    def app_factory():
        config_json = os.environ.get('DLROUTER_CONFIG_JSON')
        cfg = RouterConfig.model_validate_json(config_json) if config_json else RouterConfig()
        return create_app(cfg)

    # Run Gunicorn
    DLRouterApplication(app_factory, gunicorn_options).run()


def main():
    """CLI entry point."""
    fire.Fire(serve)


if __name__ == '__main__':
    main()
