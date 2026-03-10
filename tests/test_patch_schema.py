"""Tests for patch_schema.py."""

import pytest
import jsonschema

from patch_schema import validate_patch, PATCH_ARRAY_SCHEMA, PATCH_OPERATION_SCHEMA


# ---------------------------------------------------------------------------
# Valid patches — should not raise
# ---------------------------------------------------------------------------


def test_valid_replace():
    patch = [{"op": "replace", "path": "/foo/bar", "value": 42}]
    validate_patch(patch)  # no exception


def test_valid_add():
    patch = [{"op": "add", "path": "/items/-", "value": {"id": 99}}]
    validate_patch(patch)


def test_valid_remove():
    patch = [{"op": "remove", "path": "/scenarios/0"}]
    validate_patch(patch)


def test_valid_multiple_ops():
    patch = [
        {"op": "replace", "path": "/name", "value": "Alice"},
        {"op": "remove", "path": "/old_field"},
        {"op": "add", "path": "/new_field", "value": True},
    ]
    validate_patch(patch)


# ---------------------------------------------------------------------------
# Invalid patches — should raise ValidationError
# ---------------------------------------------------------------------------


def test_invalid_empty_array():
    with pytest.raises(jsonschema.ValidationError):
        validate_patch([])


def test_invalid_bad_op():
    patch = [{"op": "copy", "path": "/foo", "value": 1}]
    with pytest.raises(jsonschema.ValidationError):
        validate_patch(patch)


def test_invalid_missing_path():
    patch = [{"op": "replace", "value": 1}]
    with pytest.raises(jsonschema.ValidationError):
        validate_patch(patch)


def test_invalid_add_missing_value():
    # "add" requires "value" per the if/then in the schema
    patch = [{"op": "add", "path": "/foo"}]
    with pytest.raises(jsonschema.ValidationError):
        validate_patch(patch)


def test_invalid_replace_missing_value():
    patch = [{"op": "replace", "path": "/foo"}]
    with pytest.raises(jsonschema.ValidationError):
        validate_patch(patch)


def test_invalid_not_a_list():
    with pytest.raises(jsonschema.ValidationError):
        validate_patch({"op": "replace", "path": "/x", "value": 1})  # type: ignore


def test_invalid_extra_property():
    patch = [{"op": "remove", "path": "/x", "unexpected": "field"}]
    with pytest.raises(jsonschema.ValidationError):
        validate_patch(patch)
