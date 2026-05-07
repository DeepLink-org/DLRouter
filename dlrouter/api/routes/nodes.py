"""Node management routes."""

from fastapi import APIRouter, Depends

from dlrouter.api.middleware import check_api_key
from dlrouter.logger import get_logger
from dlrouter.models.node import Node


router = APIRouter(prefix='/nodes')
logger = get_logger('dlrouter.api.nodes')

# Injected by app factory
_node_manager = None


def set_node_manager(mgr) -> None:
    """Inject node manager dependency."""
    global _node_manager
    _node_manager = mgr


@router.get(
    '/status',
    dependencies=[Depends(check_api_key)],
)
async def node_status():
    """Show all registered nodes and their status."""
    try:
        return _node_manager.status
    except Exception:
        return {'error': 'Failed to get status.'}


@router.post(
    '/add',
    dependencies=[Depends(check_api_key)],
)
async def add_node(node: Node):
    """Register a new backend node.

    - **url**: HTTP URL of the inference server
    - **status**: Optional initial status info
    """
    try:
        added = _node_manager.add(node.url, node.status)
        if not added:
            return {'error': ('Failed to add node. Check the URL and try again.')}
        logger.info(f'Added node: {node.url}')
        return {'message': 'Added successfully.'}
    except Exception as e:
        logger.error(f'Add node {node.url} failed: {e}')
        return {'error': ('Failed to add node. Check the URL and try again.')}


@router.post(
    '/remove',
    dependencies=[Depends(check_api_key)],
)
async def remove_node(node: Node):
    """Remove a registered backend node."""
    try:
        _node_manager.remove(node.url)
        logger.info(f'Removed node: {node.url}')
        return {'message': 'Removed successfully.'}
    except Exception as e:
        logger.error(f'Remove node {node.url} failed: {e}')
        return {'error': 'Failed to remove node.'}


@router.post(
    '/terminate',
    dependencies=[Depends(check_api_key)],
)
async def terminate_node(node: Node):
    """Terminate and remove a backend node."""
    try:
        ok = _node_manager.terminate_node(node.url)
        if not ok:
            return {'error': (f'Failed to terminate {node.url}')}
        return {'message': 'Terminated successfully.'}
    except Exception as e:
        logger.error(f'Terminate {node.url} failed: {e}')
        return {'error': 'Failed to terminate node.'}


@router.post(
    '/terminate_all',
    dependencies=[Depends(check_api_key)],
)
async def terminate_all():
    """Terminate all registered nodes."""
    try:
        ok = _node_manager.terminate_all()
        if not ok:
            return {'error': ('Some nodes failed to terminate.')}
        return {'message': ('All nodes terminated.')}
    except Exception as e:
        logger.error(f'Terminate all failed: {e}')
        return {'error': 'Failed to terminate all nodes.'}
