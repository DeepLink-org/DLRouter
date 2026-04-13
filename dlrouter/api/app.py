"""FastAPI application factory."""

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dlrouter.api.middleware import set_api_keys
from dlrouter.api.routes import (
    chat,
    completions,
    models,
    nodes,
)
from dlrouter.backends.factory import create_backend
from dlrouter.config import RouterConfig
from dlrouter.constants import ServiceDiscoveryMode, ServingStrategy
from dlrouter.core.health_check import HealthChecker
from dlrouter.core.node_manager import NodeManager
from dlrouter.core.proxy_engine import ProxyEngine
from dlrouter.logger import get_logger


async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log full request body on validation error."""
    try:
        body = await request.body()
        body_text = body.decode('utf-8', errors='replace')
    except Exception:
        body_text = '<unable to read body>'

    logger.error(f'Validation error for {request.method} {request.url.path}. Body: {body_text}. Errors: {exc.errors()}')

    # Return standard 422 response
    return JSONResponse(
        status_code=422,
        content={'detail': exc.errors()},
    )


logger = get_logger('dlrouter.app')


def _resolve_distserve_discovery_mode(backend: Any, config: RouterConfig) -> ServiceDiscoveryMode:
    """Resolve discovery mode for DistServe startup."""
    pd_config = getattr(backend, 'pd_config', None)
    if pd_config is not None and hasattr(pd_config, 'discovery_mode'):
        return pd_config.discovery_mode

    discovery_mode_str = config.backend_config.get('discovery_mode', 'heartbeat')
    return ServiceDiscoveryMode(discovery_mode_str)


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

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        health_checker.start()
        if service_discovery:
            service_discovery.start()
        logger.info('DLRouter started.')
        yield
        health_checker.stop()
        if service_discovery:
            service_discovery.stop()
        await backend.close()
        logger.info('DLRouter stopped.')

    app = FastAPI(
        title='DLRouter',
        description=('A high-performance router for LLM inference backends'),
        version='0.1.0',
        docs_url='/',
        lifespan=lifespan,
    )

    # Register validation error handler to log full request body
    app.add_exception_handler(RequestValidationError, log_validation_error)

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
    backend = create_backend(config.backend_type, config.backend_config)

    # Node manager
    node_manager = NodeManager(
        backend=backend,
        routing_strategy=config.routing_strategy,
        serving_strategy=config.serving_strategy,
        config_path=config.config_path,
        cache_status=config.cache_status,
    )

    # Service discovery (backend-specific, e.g., ZMQ for vLLM PD mode)
    service_discovery: Optional[Any] = None
    if config.serving_strategy == ServingStrategy.DISTSERVE:
        discovery_mode = _resolve_distserve_discovery_mode(backend, config)

        service_discovery = backend.create_service_discovery(
            discovery_mode,
            config.backend_config,
            node_manager,
        )

    # Proxy engine
    proxy_engine = ProxyEngine(node_manager, service_discovery)

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
    health_checker = HealthChecker(node_manager, service_discovery)

    # Store references on app for external access
    app.state.node_manager = node_manager
    app.state.proxy_engine = proxy_engine
    app.state.health_checker = health_checker
    app.state.service_discovery = service_discovery
    app.state.config = config

    return app
