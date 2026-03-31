"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dlrouter.api.middleware import set_api_keys
from dlrouter.api.routes import (
    chat,
    completions,
    models,
    nodes,
)
from dlrouter.backends.factory import create_backend
from dlrouter.config import RouterConfig
from dlrouter.core.health_check import HealthChecker
from dlrouter.core.node_manager import NodeManager
from dlrouter.core.proxy_engine import ProxyEngine
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.app')


def create_app(
    config: RouterConfig = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Router configuration. Uses defaults
            if not provided.

    Returns:
        Configured FastAPI application.
    """
    if config is None:
        config = RouterConfig()

    app = FastAPI(
        title='DLRouter',
        description=('A high-performance router for LLM inference backends'),
        version='0.1.0',
        docs_url='/',
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # API keys
    if config.api_keys:
        set_api_keys(config.api_keys)

    # Backend
    backend = create_backend(config.backend, config.pd_config)

    # Node manager
    node_manager = NodeManager(
        backend=backend,
        routing_strategy=config.routing_strategy,
        serving_strategy=config.serving_strategy,
        config_path=config.config_path,
        cache_status=config.cache_status,
    )

    # Proxy engine
    proxy_engine = ProxyEngine(node_manager)

    # Inject dependencies into routes
    models.set_node_manager(node_manager)
    nodes.set_node_manager(node_manager)
    chat.set_dependencies(proxy_engine, node_manager)
    completions.set_dependencies(proxy_engine, node_manager)

    # Register routes
    app.include_router(models.router)
    app.include_router(nodes.router)
    app.include_router(chat.router)
    app.include_router(completions.router)

    # Health checker
    health_checker = HealthChecker(node_manager)

    @app.on_event('startup')
    async def on_startup():
        health_checker.start()
        logger.info('DLRouter started.')

    @app.on_event('shutdown')
    async def on_shutdown():
        health_checker.stop()
        # Close backend connection pool
        await backend.close()
        logger.info('DLRouter stopped.')

    # Store references on app for external access
    app.state.node_manager = node_manager
    app.state.proxy_engine = proxy_engine
    app.state.health_checker = health_checker
    app.state.config = config

    return app
