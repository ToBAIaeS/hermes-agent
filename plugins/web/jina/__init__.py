"""Jina Reader extract plugin — bundled, auto-loaded.

Backed by a user-hosted jina-reader instance (URL configured via
``JINA_READER_URL``). Extract-only — pair with a search provider
(searxng/ddgs/brave-free) for ``web_search`` calls.
"""

from __future__ import annotations

from plugins.web.jina.provider import JinaReaderWebSearchProvider


def register(ctx) -> None:
    """Register the Jina Reader provider with the plugin context."""
    ctx.register_web_search_provider(JinaReaderWebSearchProvider())
