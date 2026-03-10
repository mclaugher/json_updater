"""
patch_schema.py — JSON Schema definitions for RFC 6902 patch operations,
plus a validation helper.
"""

from __future__ import annotations

import jsonschema

# JSON Schema for a single patch operation (add / remove / replace subset).
PATCH_OPERATION_SCHEMA: dict = {
    "type": "object",
    "required": ["op", "path"],
    "properties": {
        "op": {
            "type": "string",
            "enum": ["add", "remove", "replace"],
        },
        "path": {
            "type": "string",
            "description": "JSON Pointer (RFC 6901), e.g. /foo/0/bar",
        },
        "value": {
            "description": "New value for 'add' and 'replace' operations.",
        },
    },
    "additionalProperties": False,
    "if": {
        "properties": {"op": {"enum": ["add", "replace"]}},
    },
    "then": {
        "required": ["op", "path", "value"],
    },
}

# JSON Schema for an array of patch operations — used with Ollama structured
# outputs and for local pre-application validation.
PATCH_ARRAY_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "JSONPatchArray",
    "type": "array",
    "items": PATCH_OPERATION_SCHEMA,
    "minItems": 1,
}


def validate_patch(patch: list[dict]) -> None:
    """Validate a patch array against PATCH_ARRAY_SCHEMA.

    Args:
        patch: A list of patch operation dicts.

    Raises:
        jsonschema.ValidationError: If the patch does not conform to the schema.
        jsonschema.SchemaError: If PATCH_ARRAY_SCHEMA itself is malformed
            (should never happen in normal use).
    """
    jsonschema.validate(instance=patch, schema=PATCH_ARRAY_SCHEMA)
