"""SGLang bootstrap request adapter."""

from __future__ import annotations

import ipaddress
import random
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from dlrouter.backends.utils import normalize_backend_url


if TYPE_CHECKING:
    from collections.abc import Callable


DEFAULT_BOOTSTRAP_PORT = 8998


class SGLangBootstrapAdapter:
    """Inject SGLang bootstrap fields into OpenAI-compatible requests."""

    def __init__(
        self,
        bootstrap_ports_by_url: dict[str, int | None],
        room_generator: Callable[[], int] | None = None,
    ) -> None:
        self.bootstrap_ports_by_url = {normalize_backend_url(url): port for url, port in bootstrap_ports_by_url.items()}
        self.room_generator = room_generator or _generate_bootstrap_room

    def build_request(
        self,
        request_data: dict[str, Any],
        *,
        prefill_url: str,
        endpoint: str,
    ) -> dict[str, Any]:
        """Return a copy of the request with SGLang bootstrap metadata."""
        routed_request = request_data.copy()
        batch_size = _get_request_batch_size(routed_request, endpoint)
        hostname = _bootstrap_host(prefill_url)
        bootstrap_port = self.bootstrap_ports_by_url.get(
            normalize_backend_url(prefill_url),
            DEFAULT_BOOTSTRAP_PORT,
        )

        if batch_size is not None:
            routed_request.update(
                {
                    'bootstrap_host': [hostname] * batch_size,
                    'bootstrap_port': [bootstrap_port] * batch_size,
                    'bootstrap_room': [self.room_generator() for _ in range(batch_size)],
                }
            )
            return routed_request

        routed_request.update(
            {
                'bootstrap_host': hostname,
                'bootstrap_port': bootstrap_port,
                'bootstrap_room': self.room_generator(),
            }
        )
        return routed_request


def _generate_bootstrap_room() -> int:
    """Generate a SGLang bootstrap room id."""
    return random.randint(0, 2**63 - 1)


def _get_request_batch_size(
    request_data: dict[str, Any],
    endpoint: str,
) -> int | None:
    """Infer SGLang bootstrap batch size for supported endpoints."""
    if endpoint.endswith('/v1/chat/completions'):
        n = request_data.get('n')
        if isinstance(n, int) and n > 1:
            return n
        return None

    if endpoint.endswith('/v1/completions'):
        prompt = request_data.get('prompt')
        if isinstance(prompt, list) and prompt:
            return len(prompt)
        return None

    if endpoint.endswith('/generate'):
        input_ids = request_data.get('input_ids')
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return len(input_ids)

        text = request_data.get('text')
        if isinstance(text, list):
            return len(text)

    return None


def _bootstrap_host(prefill_url: str) -> str:
    """Extract the host value SGLang expects in bootstrap_host."""
    parsed = urlparse(prefill_url)
    hostname = parsed.hostname or prefill_url
    return _maybe_wrap_ipv6_address(hostname)


def _maybe_wrap_ipv6_address(hostname: str) -> str:
    """Wrap IPv6 hosts in brackets, matching SGLang mini_lb behavior."""
    try:
        ipaddress.IPv6Address(hostname)
    except ValueError:
        return hostname
    return f'[{hostname}]'
