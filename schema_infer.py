"""
schema_infer.py — Infer and merge JSON Schemas from config files using genson.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genson import SchemaBuilder


def infer_schema(data: dict | list) -> dict:
    """Infer a JSON Schema from a Python dict or list using genson.

    Args:
        data: The Python object to infer a schema for.

    Returns:
        A JSON Schema dict.
    """
    builder = SchemaBuilder()
    builder.add_object(data)
    return builder.to_schema()


def merge_schemas(schemas: list[dict]) -> dict:
    """Merge multiple JSON Schemas into one using genson's SchemaBuilder.

    All schemas are fed into a single builder so genson handles type unions
    and property merging automatically.

    Args:
        schemas: List of JSON Schema dicts.

    Returns:
        A merged JSON Schema dict.
    """
    if not schemas:
        return {}
    if len(schemas) == 1:
        return schemas[0]

    builder = SchemaBuilder()
    for schema in schemas:
        builder.add_schema(schema)
    return builder.to_schema()


def load_and_infer(paths: list[Path]) -> dict:
    """Load JSON files from disk, infer a schema for each, and merge them.

    Args:
        paths: List of Path objects pointing to JSON files.

    Returns:
        A merged JSON Schema dict covering all loaded files.

    Raises:
        FileNotFoundError: If any path does not exist.
        json.JSONDecodeError: If any file is not valid JSON.
    """
    schemas: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        schemas.append(infer_schema(data))
    return merge_schemas(schemas)


def summarize_schema(schema: dict) -> str:
    """Produce a short human-readable description of a JSON Schema.

    Walks top-level properties (for object schemas) or item structure
    (for array schemas) and returns a concise summary suitable for
    inclusion in an LLM prompt.

    Args:
        schema: A JSON Schema dict.

    Returns:
        A multi-line string summary.
    """
    lines: list[str] = []
    schema_type = schema.get("type")

    if schema_type == "object" or "properties" in schema:
        props: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])
        lines.append(f"Top-level object with {len(props)} properties:")
        for key, sub in props.items():
            req_marker = " (required)" if key in required else ""
            sub_type = _describe_type(sub)
            lines.append(f"  - {key}: {sub_type}{req_marker}")
    elif schema_type == "array" or "items" in schema:
        items = schema.get("items", {})
        lines.append(f"Top-level array. Item schema: {_describe_type(items)}")
    elif "anyOf" in schema or "oneOf" in schema:
        variants = schema.get("anyOf", schema.get("oneOf", []))
        lines.append(f"Schema is a union of {len(variants)} types.")
        for i, v in enumerate(variants):
            lines.append(f"  Variant {i + 1}: {_describe_type(v)}")
    else:
        lines.append(f"Schema type: {schema.get('type', 'unknown')}")

    return "\n".join(lines)


def _describe_type(sub: dict) -> str:
    """Return a short string description of a sub-schema."""
    if not sub:
        return "any"
    t = sub.get("type")
    if t == "object":
        keys = list(sub.get("properties", {}).keys())
        if keys:
            preview = ", ".join(keys[:5])
            suffix = ", …" if len(keys) > 5 else ""
            return f"object({preview}{suffix})"
        return "object"
    if t == "array":
        items = sub.get("items", {})
        return f"array of {_describe_type(items)}"
    if t:
        return str(t)
    if "anyOf" in sub:
        variants = [_describe_type(v) for v in sub["anyOf"]]
        return " | ".join(variants)
    return "any"
