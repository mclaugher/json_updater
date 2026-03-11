"""Tests for context_extractor.py — path inventory and context summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so imports work when running from any dir.
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_extractor import (
    _MAX_ARRAY_ITEMS,
    _MAX_PATH_ENTRIES,
    _enumerate_paths,
    extract_context,
)


# ---------------------------------------------------------------------------
# _enumerate_paths
# ---------------------------------------------------------------------------


def test_enumerate_paths_flat_dict():
    config = {"a": 1, "b": "x"}
    result = _enumerate_paths(config)
    assert any("/a = 1" in line for line in result)
    assert any('/b = "x"' in line for line in result)


def test_enumerate_paths_nested_dict():
    config = {"project": {"name": "foo", "version": "1.0"}}
    result = _enumerate_paths(config)
    assert any('/project/name = "foo"' in line for line in result)
    assert any('/project/version = "1.0"' in line for line in result)


def test_enumerate_paths_array_limited():
    config = {"items": [{"id": i} for i in range(20)]}
    result = _enumerate_paths(config)
    # First _MAX_ARRAY_ITEMS indices should appear
    for i in range(_MAX_ARRAY_ITEMS):
        assert any(f"/items/{i}/id" in line for line in result), f"Expected /items/{i}/id"
    # Items beyond the cap should NOT appear as individual entries
    assert not any("/items/10/id" in line for line in result)
    # A truncation hint should appear for the array
    assert any("items" in line and "items" in line and "shown" in line for line in result)


def test_enumerate_paths_max_paths_cap():
    # Build a wide, flat config with more than _MAX_PATH_ENTRIES leaf keys
    config = {f"key_{i}": i for i in range(_MAX_PATH_ENTRIES + 50)}
    result = _enumerate_paths(config)
    # At most max_paths entries + 1 sentinel line
    assert len(result) <= _MAX_PATH_ENTRIES + 1
    # Last line should be the truncation sentinel
    assert "truncated" in result[-1]


def test_enumerate_paths_scalar_values_inline():
    config = {
        "start_date": "2024-01-01",
        "count": 42,
        "active": True,
    }
    result = _enumerate_paths(config)
    assert any('/start_date = "2024-01-01"' in line for line in result)
    assert any("/count = 42" in line for line in result)
    assert any("/active = true" in line for line in result)


def test_enumerate_paths_empty_config():
    assert _enumerate_paths({}) == []
    assert _enumerate_paths([]) == []


# ---------------------------------------------------------------------------
# extract_context — path inventory in context_summary
# ---------------------------------------------------------------------------


def test_context_summary_contains_path_inventory():
    """Regression: 'update the date' must surface start_date, not invent /date."""
    config = {
        "project": {
            "name": "acme",
            "start_date": "2024-01-01",
        },
        "version": 2,
    }
    _excerpt, context_summary = extract_context(config, "update the date")
    assert "start_date" in context_summary
    assert "Available JSON paths" in context_summary


def test_context_summary_always_has_inventory_even_without_keyword_match():
    """Even when no keywords match, the path inventory must still be present."""
    config = {
        "alpha": {"beta": {"gamma": "deep_value"}},
        "delta": 99,
    }
    # Instruction with no substring overlap with any key
    _excerpt, context_summary = extract_context(config, "change the zzzzzz thing")
    assert "Available JSON paths" in context_summary
    assert "/alpha/beta/gamma" in context_summary


def test_context_summary_nested_fields_visible():
    """Fields buried 3 levels deep appear in the inventory."""
    config = {
        "schedule": {
            "recurring": {
                "end_date": "2025-12-31",
            }
        }
    }
    _excerpt, context_summary = extract_context(config, "update the end date")
    assert "/schedule/recurring/end_date" in context_summary
