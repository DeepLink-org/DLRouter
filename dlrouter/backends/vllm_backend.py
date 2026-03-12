"""vLLM backend adapter.

Supports standard OpenAI-compatible API forwarding
for vLLM inference engine.
"""

from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import requests

from dlrouter.backends.base import BaseBackend
from dlrouter.constants import AIOHTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT
from dlrouter.logger import get_logger


logger = get_logger('dlrouter.backends.vllm')


class VLLMBackend(BaseBackend):
    """Backend adapter for vLLM inference engine.

    Handles standard OpenAI-compatible API forwarding.
    vLLM does not support PD disaggregation.
    """

    def __init__(self) -> None:
        timeout_val = AIOHTTP_TIMEOUT
        self._timeout = aiohttp.ClientTimeout(total=timeout_val)

    # -- Core forwarding --

    async def forward_request(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
        stream: bool = False,
    ) -> Any:
        """Forward request to vLLM node."""
        try:
            async with aiohttp.ClientSession() as sess:
                url = node_url + endpoint
                async with sess.post(
                    url,
                    json=request_data,
                    timeout=self._timeout,
                ) as resp:
                    return await resp.text()
        except Exception as e:
            logger.error(f'Forward error: {e}')
            raise

    async def stream_forward(
        self,
        node_url: str,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Stream-forward request to vLLM node."""
        try:
            async with aiohttp.ClientSession() as sess:
                url = node_url + endpoint
                async with sess.post(
                    url,
                    json=request_data,
                    timeout=self._timeout,
                ) as resp:
                    async for line in resp.content:
                        if line.strip():
                            yield line + b'\n\n'
        except Exception as e:
            logger.error(f'Stream error: {e}')
            raise

    def fetch_models(self, node_url: str) -> list[str]:
        """Fetch available models from vLLM node."""
        try:
            url = f'{node_url}/v1/models'
            headers = {'accept': 'application/json'}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            models = [m['id'] for m in data.get('data', [])]
            return models
        except Exception as e:
            logger.error(f'Failed to fetch models from {node_url}: {e}')
            return []

    async def check_health(self, node_url: str) -> bool:
        """Check vLLM node health via async request."""
        try:
            url = f'{node_url}/health'
            timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            async with aiohttp.ClientSession() as sess, sess.get(url, timeout=timeout) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f'Failed to check health from {node_url}: {e}')
            return False

    def deregister_node(self, node_url: str) -> None:
        """No-op for vLLM (no PD connection pool)."""
