"""Gunicorn configuration for DLRouter.

This configuration enables multi-process mode with Uvicorn workers,
allowing DLRouter to utilize multiple CPU cores for better performance.

Usage:
    gunicorn dlrouter.api.app:create_app \
        --config dlrouter/gunicorn_conf.py \
        --bind 0.0.0.0:8000

Or with the CLI:
    dlrouter serve --workers 4
"""

import multiprocessing
import os


# Worker configuration
workers = int(os.getenv('DLROUTER_WORKERS', multiprocessing.cpu_count()))
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = int(os.getenv('DLROUTER_WORKER_CONNECTIONS', 1000))

# Timeouts
timeout = int(os.getenv('DLROUTER_TIMEOUT', 120))
graceful_timeout = int(os.getenv('DLROUTER_GRACEFUL_TIMEOUT', 30))
keepalive = int(os.getenv('DLROUTER_KEEPALIVE', 5))

# Server socket
bind = os.getenv('DLROUTER_BIND', '0.0.0.0:8000')
backlog = int(os.getenv('DLROUTER_BACKLOG', 2048))

# Process naming
proc_name = 'dlrouter'

# Logging
loglevel = os.getenv('DLROUTER_LOG_LEVEL', 'info').lower()
accesslog = os.getenv('DLROUTER_ACCESS_LOG', '-')
errorlog = os.getenv('DLROUTER_ERROR_LOG', '-')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Preload app for memory efficiency (shared memory across workers)
# NOTE: Set to False if you need per-worker state isolation
preload_app = os.getenv('DLROUTER_PRELOAD', 'false').lower() == 'true'

# Max requests per worker (helps prevent memory leaks)
max_requests = int(os.getenv('DLROUTER_MAX_REQUESTS', 10000))
max_requests_jitter = int(os.getenv('DLROUTER_MAX_REQUESTS_JITTER', 1000))


def on_starting(server):
    """Called just before the master process is initialized."""
    pass


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    pass


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    pass


def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    pass
