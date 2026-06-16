"""Jina Reader — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. URL-to-
markdown extraction via a user-hosted jina-reader instance (the
`jinaai/readermf` Docker image is Apache 2.0 and works fine self-hosted).

Extract-only — jina-reader does not implement a search endpoint; pair with
a search provider (``searxng``, ``ddgs``, ``brave-free``, ...) for
``web_search`` calls. ``supports_search()`` returns False.

Config keys this provider responds to::

    web:
      extract_backend: "jina"     # explicit per-capability
      backend: "jina"             # shared fallback

Env var::

    JINA_READER_URL=http://localhost:3001
    # Optional: token-based auth if your instance requires it
    JINA_READER_API_KEY=...
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class JinaReaderWebSearchProvider(WebSearchProvider):
    """URL → markdown extraction via a user-hosted jina-reader instance."""

    # Per-URL HTTP timeout. Jina typically returns in 5-15s, but a slow
    # target site can push it past 30s.
    _EXTRACT_TIMEOUT_S = 60.0

    @property
    def name(self) -> str:
        return "jina"

    @property
    def display_name(self) -> str:
        return "Jina Reader (self-hosted)"

    def is_available(self) -> bool:
        """Return True when ``JINA_READER_URL`` is set.

        Per the WebSearchProvider contract, this must NOT make a network
        call — it's invoked at tool-registration time and on every
        ``hermes tools`` paint.
        """
        return bool(os.getenv("JINA_READER_URL", "").strip())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(
        self,
        urls: List[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Extract markdown content from one or more URLs.

        jina-reader's HTTP API is GET-based: append the target URL to
        ``JINA_READER_URL`` and it returns markdown as text/plain. The
        ``X-Return-Format: markdown`` header is the default; we also pass
        ``X-Timeout`` so the upstream site can't hang us past
        :pyattr:`_EXTRACT_TIMEOUT_S`.

        Returns a list of dicts in the shape documented on
        :meth:`WebSearchProvider.extract` so the dispatcher can pass them
        straight through to ``web_extract_tool``'s post-processing pipeline.
        Per-URL failures are reported as ``{"error": ...}`` entries — we
        never raise out of this method.
        """
        base_url = os.getenv("JINA_READER_URL", "").strip().rstrip("/")
        api_key = os.getenv("JINA_READER_API_KEY", "").strip()

        if not base_url:
            # Should be gated by is_available(), but be defensive in case
            # the env var was unset between is_available() and extract().
            return [
                {"url": u, "error": "JINA_READER_URL is not set"}
                for u in urls
            ]

        # Optional forward-compat kwargs the dispatcher may pass through.
        # jina-reader uses its own markdown output by default, so we
        # ignore `format` and `include_raw` for now.
        _ = kwargs  # silence unused-arg linters

        headers: Dict[str, str] = {
            "Accept": "text/plain",
            # NOTE: jina-reader's own X-Timeout header tells the upstream
            # site-fetch how long to wait. We don't set it here — let
            # jina use its own default (60s). Our outer httpx timeout
            # below is the real backstop.
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        results: List[Dict[str, Any]] = []
        for url in urls:
            entry: Dict[str, Any] = {"url": url}
            try:
                # jina expects the target URL *as the path*, not as a
                # query param. Strip a leading slash from the target so we
                # don't end up with `//` in the joined URL.
                target = url.lstrip("/")
                full_url = f"{base_url}/{target}"
                resp = httpx.get(
                    full_url,
                    headers=headers,
                    timeout=self._EXTRACT_TIMEOUT_S,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                body = resp.text
                # jina returns the markdown body as the response text.
                # The legacy extract pipeline uses `content` for the
                # processed view and `raw_content` for the raw payload;
                # for a markdown-only extractor the two are identical.
                entry["content"] = body
                entry["raw_content"] = body
                entry["metadata"] = {
                    "status_code": resp.status_code,
                    "content_type": resp.headers.get("content-type", ""),
                    "extractor": "jina-reader",
                }
                # jina returns a header line "Title: <title>" followed
                # by "URL Source: <url>" then "Markdown Content:\n<body>".
                # Parse that structured prefix rather than scanning for a
                # markdown H1, which jina doesn't always emit.
                title = url
                content = body
                title_marker = "Title:"
                url_marker = "URL Source:"
                content_marker = "Markdown Content:"
                if title_marker in body and url_marker in body and content_marker in body:
                    try:
                        after_title = body.split(title_marker, 1)[1]
                        title_line = after_title.split("\n", 1)[0].strip()
                        if title_line:
                            title = title_line
                    except Exception:
                        pass
                entry["title"] = title
            except httpx.HTTPStatusError as exc:
                logger.warning("jina HTTP %s for %s", exc.response.status_code, url)
                entry["error"] = f"jina HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            except httpx.RequestError as exc:
                logger.warning("jina request error for %s: %s", url, exc)
                entry["error"] = f"jina request error: {type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - never let one URL fail the batch
                logger.warning("jina unexpected error for %s: %s", url, exc)
                entry["error"] = f"jina unexpected error: {type(exc).__name__}: {exc}"
            results.append(entry)

        return results
