"""Text completions route."""

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from dlrouter.api.middleware import check_api_key
from dlrouter.models.protocol import CompletionRequest


router = APIRouter()

# Injected by app factory
_proxy_engine = None
_node_manager = None


def set_dependencies(proxy_engine, node_manager):
    """Inject dependencies."""
    global _proxy_engine, _node_manager
    _proxy_engine = proxy_engine
    _node_manager = node_manager


@router.post(
    '/v1/completions',
    dependencies=[Depends(check_api_key)],
)
async def completions(
    request: CompletionRequest,
    raw_request: Request = None,
):
    """Text completion API (OpenAI-compatible).

    Refer to OpenAI API specification:
    https://platform.openai.com/docs/api-reference/completions

    Fields:
    - **model**: model name (from /v1/models)
    - **prompt**: input prompt string or list
    - **temperature**: sampling temperature
    - **top_p**: nucleus sampling threshold
    - **n**: number of choices (only 1 supported)
    - **stream**: whether to stream response
    - **max_tokens**: max output tokens
    - **stop**: stop sequences
    """
    model = request.model
    if model not in _node_manager.model_list:
        return JSONResponse(
            status_code=HTTPStatus.NOT_FOUND,
            content={
                'error': {
                    'message': (f'Model {model!r} not found.'),
                    'type': 'invalid_request_error',
                    'code': 'model_not_found',
                }
            },
        )

    request_data = request.model_dump(exclude_none=True)

    return await _proxy_engine.dispatch(
        request_data=request_data,
        model_name=model,
        endpoint='/v1/completions',
        stream=bool(request.stream),
        raw_request=raw_request,
        body=request,
    )
