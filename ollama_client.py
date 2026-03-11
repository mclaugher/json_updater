"""
ollama_client.py — Wrapper around the Ollama HTTP API for structured JSON output.

Uses Ollama's native structured-output support: the ``format`` field in the
/api/chat payload accepts a JSON Schema object directly (Ollama ≥ 0.5).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "gemma2:9b"
_MAX_RETRIES = 2
_RETRY_DELAY = 1.5  # seconds between retries

# Keys that Ollama's format parser rejects — strip them from the schema copy
# before sending.  We never mutate the caller's original dict.
_SCHEMA_STRIP_KEYS = {"$schema", "$id", "title", "description"}


class OllamaError(Exception):
    """Raised when an Ollama API call fails after all retries."""


class OllamaClient:
    """Thin wrapper around Ollama's /api/chat endpoint.

    Sends a two-message conversation (system + user) and asks Ollama to
    constrain its output to a caller-supplied JSON Schema via the ``format``
    parameter.  The response content is JSON-decoded and returned as a Python
    object.

    Args:
        model: Name of the Ollama model to use (e.g. ``"gemma2:9b"``).
        host: Base URL of the Ollama server (default: ``"http://localhost:11434"``).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the Ollama server is reachable.

        Calls GET /api/tags (a lightweight endpoint that lists local models).
        Does not raise; returns False on any error so callers can give a
        friendly message before the first real call.
        """
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        system: str,
        user: str,
        format_schema: dict,
    ) -> Any:
        """Send a chat request to Ollama and return the parsed JSON response.

        Args:
            system: System prompt text.
            user: User message text.
            format_schema: A JSON Schema dict used as Ollama's ``format``
                parameter to enforce structured output.  Top-level meta-keys
                (``$schema``, ``$id``, etc.) are stripped from the copy sent
                to Ollama because some versions reject them.

        Returns:
            The parsed Python object produced by the model (list or dict
            matching ``format_schema``).

        Raises:
            OllamaError: If the request fails or the response cannot be
                parsed after all retries.
        """
        clean_schema = _strip_meta_keys(format_schema)
        payload = {
            "model": self.model,
            "messages": self._build_messages(system, user),
            "stream": False,
            "format": clean_schema,
            "options": {
                # Keep temperature low for deterministic patch generation.
                "temperature": 0.1,
            },
        }

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 2):  # 1 initial + _MAX_RETRIES retries
            try:
                logger.debug(
                    "Ollama /api/chat — attempt %d/%d  model=%s",
                    attempt,
                    _MAX_RETRIES + 1,
                    self.model,
                )
                response = self._session.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                body = response.json()
                content: str = body["message"]["content"]
                logger.debug("Ollama raw response: %s", content[:300])
                return json.loads(content)

            except requests.HTTPError as exc:
                # Surface the response body for diagnosis (model not found, etc.)
                detail = ""
                try:
                    detail = exc.response.json().get("error", "")
                except Exception:
                    pass
                last_exc = OllamaError(
                    f"HTTP {exc.response.status_code}: {detail or exc}"
                )
                logger.warning("Ollama attempt %d HTTP error: %s", attempt, last_exc)

            except requests.ConnectionError as exc:
                last_exc = OllamaError(
                    f"Cannot connect to Ollama at {self.host} — is 'ollama serve' running?"
                )
                logger.warning("Ollama attempt %d connection error: %s", attempt, exc)

            except (KeyError, json.JSONDecodeError) as exc:
                last_exc = OllamaError(f"Unexpected response format: {exc}")
                logger.warning("Ollama attempt %d parse error: %s", attempt, exc)

            except requests.RequestException as exc:
                last_exc = OllamaError(str(exc))
                logger.warning("Ollama attempt %d request error: %s", attempt, exc)

            if attempt <= _MAX_RETRIES:
                logger.info("Retrying in %.1fs…", _RETRY_DELAY)
                time.sleep(_RETRY_DELAY)

        raise last_exc or OllamaError("Unknown failure")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, system: str, user: str) -> list[dict]:
        """Format system + user strings into the Ollama messages list.

        Args:
            system: System prompt.
            user: User message.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _strip_meta_keys(schema: dict) -> dict:
    """Return a shallow copy of *schema* with Ollama-incompatible keys removed.

    Only strips top-level keys; nested ``$schema`` references inside ``items``
    or ``properties`` are harmless and left in place.
    """
    return {k: v for k, v in schema.items() if k not in _SCHEMA_STRIP_KEYS}
