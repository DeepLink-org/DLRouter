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

    # With LMDeploy PD disaggregation
    python -m dlrouter --serving_strategy distserve --backend lmdeploy \
        --migration_protocol RDMA --link_type RoCE

    # With vLLM PD disaggregation
    python -m dlrouter --serving_strategy distserve --backend vllm \
        --zmq_port 30001 --models "model-a,model-b"
"""

import argparse
import os
import sys
from typing import Any

import uvicorn

from dlrouter.api.app import create_app
from dlrouter.backends.factory import get_backend_definition
from dlrouter.config import RouterConfig, SSLConfig
from dlrouter.constants import (
    BackendType,
    RoutingStrategy,
    ServingStrategy,
)
from dlrouter.logger import get_logger


logger = get_logger('dlrouter')


def build_base_parser() -> argparse.ArgumentParser:
    """Build base argument parser with common options."""
    parser = argparse.ArgumentParser(
        description='DLRouter - High-performance LLM inference router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # Disable auto help to handle it manually
    )

    # Help option (manual)
    parser.add_argument(
        '-h',
        '--help',
        action='store_true',
        help='Show this help message and exit',
    )

    # Server options
    parser.add_argument(
        '--server_name',
        default='0.0.0.0',
        help='Bind address (default: 0.0.0.0)',
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=8000,
        help='Listen port (default: 8000)',
    )

    # Backend and strategy
    parser.add_argument(
        '--backend',
        choices=[b.value for b in BackendType],
        default='lmdeploy',
        help='Inference backend type (default: lmdeploy)',
    )
    parser.add_argument(
        '--routing_strategy',
        choices=[r.value for r in RoutingStrategy],
        default='min_expected_latency',
        help='Request routing strategy (default: min_expected_latency)',
    )
    parser.add_argument(
        '--serving_strategy',
        choices=[s.value for s in ServingStrategy],
        default='hybrid',
        help='Serving mode (default: hybrid)',
    )

    # Auth and SSL
    parser.add_argument(
        '--api_keys',
        default=None,
        help='Comma-separated API keys for authentication',
    )
    parser.add_argument(
        '--ssl',
        action='store_true',
        help='Enable SSL (requires SSL_KEYFILE and SSL_CERTFILE env vars)',
    )

    # Logging and misc
    parser.add_argument(
        '--log_level',
        default='INFO',
        help='Logging level (default: INFO)',
    )
    parser.add_argument(
        '--disable_cache_status',
        action='store_true',
        help='Disable config caching',
    )
    parser.add_argument(
        '--config_path',
        default=None,
        help='Path to config persistence file',
    )

    # Multi-process
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1, use >1 for gunicorn)',
    )

    return parser


def add_backend_args(
    parser: argparse.ArgumentParser,
    backend_type: str,
) -> None:
    """Dynamically add backend-specific arguments.

    Args:
        parser: The argument parser to add args to.
        backend_type: Backend type string (e.g., 'lmdeploy', 'vllm').
    """
    try:
        definition = get_backend_definition(BackendType(backend_type))
    except ValueError:
        return  # Unknown backend, skip

    backend_group = parser.add_argument_group(
        f'{backend_type.upper()} options',
    )

    for arg in definition.get_cli_args():
        kwargs: dict[str, Any] = {
            'default': arg.default,
            'help': arg.help,
        }

        if arg.type is bool:
            # Boolean args use store_true
            if arg.default is False:
                kwargs['action'] = 'store_true'
                kwargs.pop('default', None)
            else:
                # For default=True, use --no-xxx to disable
                backend_group.add_argument(
                    f'--no_{arg.name}',
                    dest=arg.name,
                    action='store_false',
                    help=f'Disable {arg.name}',
                )
                continue
        else:
            kwargs['type'] = arg.type
            if arg.choices:
                kwargs['choices'] = arg.choices

        backend_group.add_argument(f'--{arg.name}', **kwargs)


def extract_backend_config(
    args: argparse.Namespace,
    backend_type: str,
) -> dict[str, Any]:
    """Extract backend-specific config from parsed args.

    Args:
        args: Parsed argument namespace.
        backend_type: Backend type string.

    Returns:
        Dict of backend-specific configuration.
    """
    try:
        definition = get_backend_definition(BackendType(backend_type))
    except ValueError:
        return {}

    backend_arg_names = [a.name for a in definition.get_cli_args()]
    return {k: getattr(args, k) for k in backend_arg_names if hasattr(args, k)}


def serve(
    args: argparse.Namespace,
    backend_config: dict[str, Any],
) -> None:
    """Launch the DLRouter proxy server.

    Args:
        args: Parsed CLI arguments.
        backend_config: Backend-specific configuration dict.
    """
    # Parse api_keys
    api_keys = None
    if args.api_keys:
        api_keys = [k.strip() for k in args.api_keys.split(',')]

    # Build config
    config = RouterConfig(
        server_name=args.server_name,
        server_port=args.server_port,
        routing_strategy=RoutingStrategy(args.routing_strategy),
        serving_strategy=ServingStrategy(args.serving_strategy),
        backend_type=BackendType(args.backend),
        backend_config=backend_config,
        ssl=SSLConfig(enabled=args.ssl),
        api_keys=api_keys,
        log_level=args.log_level,
        cache_status=not args.disable_cache_status,
        config_path=args.config_path,
    )

    # Set log level
    logger.setLevel(args.log_level.upper())

    # Multi-process mode with Gunicorn
    if args.workers > 1:
        _run_with_gunicorn(
            server_name=args.server_name,
            server_port=args.server_port,
            workers=args.workers,
            log_level=args.log_level,
            ssl=args.ssl,
            config=config,
        )
        return

    # Single-process mode with Uvicorn
    app = create_app(config)

    # SSL
    ssl_keyfile = None
    ssl_certfile = None
    if args.ssl:
        ssl_keyfile = os.environ.get('SSL_KEYFILE')
        ssl_certfile = os.environ.get('SSL_CERTFILE')

    # Launch
    uv_log = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(
        app=app,
        host=args.server_name,
        port=args.server_port,
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
    """CLI entry point with two-phase argument parsing.

    Phase 1: Parse base arguments to determine backend type.
    Phase 2: Add backend-specific arguments and re-parse.
    """
    import sys

    # Phase 1: Parse known args to get backend type
    parser = build_base_parser()
    args, _ = parser.parse_known_args()

    # Phase 2: Add backend-specific args
    add_backend_args(parser, args.backend)

    # If help was requested, show help now (with backend args)
    if args.help:
        parser.print_help()
        sys.exit(0)

    # Full parse
    args = parser.parse_args()

    # Extract backend config
    backend_config = extract_backend_config(args, args.backend)

    # Log startup info
    logger.info(
        f'Starting DLRouter: backend={args.backend}, '
        f'serving_strategy={args.serving_strategy}, '
        f'routing_strategy={args.routing_strategy}',
    )
    if backend_config:
        logger.info(f'Backend config: {backend_config}')

    # Launch server
    serve(args, backend_config)


if __name__ == '__main__':
    main()
