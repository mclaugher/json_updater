"""Tests for schema_infer.py."""

import json
import tempfile
from pathlib import Path

import pytest

from schema_infer import infer_schema, load_and_infer, merge_schemas, summarize_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_OBJECT = {
    "name": "Alice",
    "age": 30,
    "active": True,
    "scores": [1, 2, 3],
}

NESTED_OBJECT = {
    "project": {"title": "foo", "version": "1.0"},
    "items": [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}],
}

ANOTHER_OBJECT = {
    "name": "Bob",
    "age": 25,
    "active": False,
    "extra": "only-in-second",
}


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------


def test_infer_schema_simple_object():
    schema = infer_schema(SIMPLE_OBJECT)
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "name" in props
    assert props["name"]["type"] == "string"
    assert "age" in props
    assert "active" in props
    assert "scores" in props
    assert props["scores"]["type"] == "array"


def test_infer_schema_array():
    data = [{"id": 1, "val": "x"}, {"id": 2, "val": "y"}]
    schema = infer_schema(data)
    assert schema["type"] == "array"


def test_infer_schema_nested():
    schema = infer_schema(NESTED_OBJECT)
    assert schema["type"] == "object"
    assert "project" in schema["properties"]
    assert "items" in schema["properties"]


# ---------------------------------------------------------------------------
# merge_schemas
# ---------------------------------------------------------------------------


def test_merge_schemas_empty():
    assert merge_schemas([]) == {}


def test_merge_schemas_single():
    s = infer_schema(SIMPLE_OBJECT)
    assert merge_schemas([s]) == s


def test_merge_schemas_two_objects():
    s1 = infer_schema(SIMPLE_OBJECT)
    s2 = infer_schema(ANOTHER_OBJECT)
    merged = merge_schemas([s1, s2])
    # Both have "name", "age", "active"; merged should contain all
    # (genson uses anyOf or properties union)
    schema_str = json.dumps(merged)
    assert "name" in schema_str
    assert "extra" in schema_str


# ---------------------------------------------------------------------------
# load_and_infer
# ---------------------------------------------------------------------------


def test_load_and_infer_single_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(SIMPLE_OBJECT, f)
        tmp_path = Path(f.name)

    try:
        schema = load_and_infer([tmp_path])
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
    finally:
        tmp_path.unlink()


def test_load_and_infer_multiple_files():
    with (
        tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f1,
        tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f2,
    ):
        json.dump(SIMPLE_OBJECT, f1)
        json.dump(ANOTHER_OBJECT, f2)
        p1, p2 = Path(f1.name), Path(f2.name)

    try:
        schema = load_and_infer([p1, p2])
        schema_str = json.dumps(schema)
        assert "name" in schema_str
    finally:
        p1.unlink()
        p2.unlink()


# ---------------------------------------------------------------------------
# summarize_schema
# ---------------------------------------------------------------------------


def test_summarize_schema_object():
    schema = infer_schema(SIMPLE_OBJECT)
    summary = summarize_schema(schema)
    assert "name" in summary
    assert "age" in summary


def test_summarize_schema_array():
    schema = infer_schema([{"id": 1}])
    summary = summarize_schema(schema)
    assert "array" in summary.lower()


def test_summarize_schema_empty():
    summary = summarize_schema({})
    assert isinstance(summary, str)
