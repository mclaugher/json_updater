"""
context_extractor.py — Extract a small, relevant JSON excerpt and a
context summary from a large config dict, given a natural-language instruction.
"""

from __future__ import annotations

import json
import re
from typing import Any

from schema_infer import infer_schema, summarize_schema

# Maximum number of JSON lines to include in the excerpt sent to the model.
_MAX_EXCERPT_LINES = 50


def extract_context(
    config: dict | list,
    instruction: str,
) -> tuple[dict | list, str]:
    """Select a relevant JSON excerpt and produce a context summary.

    Strategy:
    1. Tokenise the instruction into keywords (lower-case alpha tokens).
    2. Walk the top-level keys of *config* (or indices for arrays) and collect
       subtrees whose key/index string matches any keyword.
    3. If nothing matches, fall back to the first top-level subtree(s) up to
       the line budget.
    4. Build ``context_summary`` from the inferred schema of the *full* config.

    Args:
        config: The full loaded config (dict or list).
        instruction: The analyst's natural-language instruction.

    Returns:
        A 2-tuple ``(json_excerpt, context_summary)``.
        ``json_excerpt`` is a dict or list with matched subtrees.
        ``context_summary`` is a short string description.
    """
    keywords = _tokenize(instruction)
    excerpt = _select_excerpt(config, keywords)
    schema = infer_schema(config)
    summary = summarize_schema(schema)
    context_summary = _build_context_summary(config, summary)
    return excerpt, context_summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Return lower-case alphabetic tokens from *text* (length >= 3)."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)}


def _select_excerpt(config: dict | list, keywords: set[str]) -> dict | list:
    """Return a trimmed excerpt from *config* using keyword matching."""
    if isinstance(config, dict):
        return _excerpt_from_dict(config, keywords)
    if isinstance(config, list):
        return _excerpt_from_list(config, keywords)
    # Scalar config — just return it
    return config  # type: ignore[return-value]


def _excerpt_from_dict(config: dict, keywords: set[str]) -> dict:
    """Select matching top-level keys, falling back to first keys."""
    matched: dict = {}

    # First pass: direct keyword match on key names
    for key, value in config.items():
        if _key_matches(key, keywords):
            matched[key] = value
            if _json_lines(matched) >= _MAX_EXCERPT_LINES:
                break

    # Also look one level deeper for keyword matches
    if not matched:
        for key, value in config.items():
            if isinstance(value, dict):
                sub_match = {
                    k: v for k, v in value.items() if _key_matches(k, keywords)
                }
                if sub_match:
                    matched[key] = sub_match
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Check first element keys
                sample = value[0]
                if any(_key_matches(k, keywords) for k in sample):
                    matched[key] = value[:3]  # first 3 items as sample
            if _json_lines(matched) >= _MAX_EXCERPT_LINES:
                break

    # Fallback: include first few top-level keys
    if not matched:
        for key, value in config.items():
            matched[key] = value
            if _json_lines(matched) >= _MAX_EXCERPT_LINES:
                break

    return _trim_to_budget(matched)


def _excerpt_from_list(config: list, keywords: set[str]) -> list:
    """Select matching items from a top-level array."""
    matched: list = []

    if config and isinstance(config[0], dict):
        for item in config:
            if any(_key_matches(k, keywords) for k in item):
                matched.append(item)
                if _json_lines(matched) >= _MAX_EXCERPT_LINES:
                    break

    if not matched:
        matched = config[:5]

    return _trim_list_to_budget(matched)


def _key_matches(key: str, keywords: set[str]) -> bool:
    """Return True if any keyword appears in *key* (case-insensitive)."""
    key_lower = key.lower()
    return any(kw in key_lower for kw in keywords)


def _json_lines(obj: Any) -> int:
    """Return approximate line count of JSON-serialised *obj*."""
    return len(json.dumps(obj, indent=2).splitlines())


def _trim_to_budget(obj: dict) -> dict:
    """Remove entries from *obj* until it fits within _MAX_EXCERPT_LINES."""
    keys = list(obj.keys())
    while _json_lines(obj) > _MAX_EXCERPT_LINES and keys:
        del obj[keys.pop()]
    return obj


def _trim_list_to_budget(items: list) -> list:
    """Trim *items* list until it fits within _MAX_EXCERPT_LINES."""
    while _json_lines(items) > _MAX_EXCERPT_LINES and items:
        items = items[:-1]
    return items


def _build_context_summary(config: dict | list, schema_summary: str) -> str:
    """Combine schema summary with top-level key listing."""
    lines = [schema_summary, ""]
    if isinstance(config, dict):
        keys = list(config.keys())
        lines.append(f"Top-level keys: {', '.join(keys)}")
    elif isinstance(config, list):
        lines.append(f"Top-level array with {len(config)} items.")
        if config and isinstance(config[0], dict):
            sample_keys = list(config[0].keys())
            lines.append(f"Item keys (sample): {', '.join(sample_keys)}")
    return "\n".join(lines)
