"""Health and model listing routes."""

from fastapi import APIRouter, Depends

from dlrouter.api.middleware import check_api_key
from dlrouter.models.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)


router = APIRouter()

# Will be set by app factory
_node_manager = None


def set_node_manager(mgr) -> None:
    """Inject node manager dependency."""
    global _node_manager
    _node_manager = mgr


@router.get('/health')
async def health():
    """Health check endpoint."""
    return {'status': 'ok'}


@router.get(
    '/v1/models',
    dependencies=[Depends(check_api_key)],
)
async def list_models():
    """List available models across all nodes."""
    cards = []
    for name in _node_manager.model_list:
        cards.append(
            ModelCard(
                id=name,
                root=name,
                permission=[ModelPermission()],
            )
        )
    return ModelList(data=cards)
