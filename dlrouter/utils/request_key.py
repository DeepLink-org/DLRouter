"""Request key extraction utilities for consistent hash routing.

This module implements request key extraction following vLLM router's
priority scheme for session affinity routing.

Priority Order:
1. HTTP Header: x-session-id
2. HTTP Header: x-user-id
3. HTTP Header: x-tenant-id
4. HTTP Header: x-request-id
5. HTTP Header: x-correlation-id
6. HTTP Header: x-trace-id
7. Request Body: session_params.session_id
8. Request Body: user (OpenAI format)
9. Request Body: session_id (legacy)
10. Request Body: user_id (legacy)
11. Fallback: Hash of request body
"""

import hashlib
import json
from typing import Any, Optional, Union

from starlette.requests import Request

from dlrouter.models.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


# Header keys in priority order
REQUEST_KEY_HEADERS = [
    'x-session-id',
    'x-user-id',
    'x-tenant-id',
    'x-request-id',
    'x-correlation-id',
    'x-trace-id',
]


def _hash_body(body: dict[str, Any]) -> str:
    """Generate a deterministic hash of the request body.

    Args:
        body: Request body dictionary.

    Returns:
        MD5 hash string of the sorted JSON representation.
    """
    # Sort keys for deterministic hashing
    serialized = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(serialized.encode()).hexdigest()


def extract_request_key_from_headers(request: Request) -> Optional[str]:
    """Extract request key from HTTP headers.

    Checks headers in priority order and returns the first
    non-empty value found.

    Args:
        request: FastAPI/Starlette Request object.

    Returns:
        Header value if found, None otherwise.
    """
    if request is None:
        return None

    for header in REQUEST_KEY_HEADERS:
        value = request.headers.get(header)
        if value:
            return value
    return None


def extract_request_key_from_body(
    body: Union[ChatCompletionRequest, CompletionRequest, dict[str, Any]],
) -> Optional[str]:
    """Extract request key from request body.

    Checks body fields in priority order:
    1. session_params.session_id
    2. user (OpenAI format)
    3. session_id (legacy)
    4. user_id (legacy)

    Args:
        body: Parsed request body (Pydantic model or dict).

    Returns:
        Body field value if found, None otherwise.
    """
    # Handle Pydantic models
    if isinstance(body, (ChatCompletionRequest, CompletionRequest)):
        # Check session_params.session_id
        if body.session_params and body.session_params.session_id:
            return body.session_params.session_id
        # Check user (OpenAI format)
        if body.user:
            return body.user
        # Check legacy fields
        if body.session_id:
            return body.session_id
        if body.user_id:
            return body.user_id
        return None

    # Handle dict
    if isinstance(body, dict):
        # Check session_params.session_id
        session_params = body.get('session_params')
        if isinstance(session_params, dict):
            session_id = session_params.get('session_id')
            if session_id:
                return session_id
        # Check user (OpenAI format)
        if body.get('user'):
            return body['user']
        # Check legacy fields
        if body.get('session_id'):
            return body['session_id']
        if body.get('user_id'):
            return body['user_id']

    return None


def extract_request_key(
    request: Optional[Request] = None,
    body: Union[ChatCompletionRequest, CompletionRequest, dict[str, Any], None] = None,
    fallback_to_hash: bool = True,
) -> Optional[str]:
    """Extract request key for consistent hash routing.

    Implements the full priority chain:
    1-6. HTTP Headers (x-session-id, x-user-id, etc.)
    7-10. Request body fields
    11. Fallback to body hash

    Args:
        request: FastAPI/Starlette Request object (for headers).
        body: Parsed request body (Pydantic model or dict).
        fallback_to_hash: If True, compute hash of body when no
            explicit key is found.

    Returns:
        Request key string, or None if no key found and
        fallback is disabled.

    Example:
        >>> from fastapi import Request
        >>> from dlrouter.models.protocol import ChatCompletionRequest
        >>>
        >>> async def handler(request: Request, body: ChatCompletionRequest):
        ...     key = extract_request_key(request, body)
        ...     # Use key for consistent hash routing
    """
    # Priority 1-6: Check headers
    header_key = extract_request_key_from_headers(request)
    if header_key:
        return header_key

    # Priority 7-10: Check body fields
    body_key = extract_request_key_from_body(body)
    if body_key:
        return body_key

    # Priority 11: Fallback to body hash
    if fallback_to_hash and body is not None:
        if isinstance(body, (ChatCompletionRequest, CompletionRequest)):
            body_dict = body.model_dump(exclude_none=True)
        elif isinstance(body, dict):
            body_dict = body
        else:
            return None
        return _hash_body(body_dict)

    return None
