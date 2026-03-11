"""Tests for patch_engine.py — uses a stubbed OllamaClient."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from patch_engine import ApplyResult, PatchEngine, _deep_copy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "project": {"name": "test-project", "version": "1.0"},
    "scenarios": [
        {"id": "s1", "name": "Alpha", "enabled": True, "value": 10},
        {"id": "s2", "name": "Beta", "enabled": False, "value": 20},
    ],
}

VALID_PATCH = [{"op": "replace", "path": "/scenarios/0/value", "value": 99}]
REMOVE_PATCH = [{"op": "remove", "path": "/scenarios/1"}]
ADD_PATCH = [
    {"op": "add", "path": "/scenarios/-", "value": {"id": "s3", "name": "Gamma", "enabled": True, "value": 30}}
]


def _make_engine(config: dict, patch_response: list[dict]) -> PatchEngine:
    """Create a PatchEngine with an in-memory temp config and stubbed Ollama."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(config, f)
        tmp_path = Path(f.name)

    engine = PatchEngine(
        config_paths=[tmp_path],
        model="gemma2:9b",
        host="http://localhost:11434",
        project_dir=Path(tempfile.gettempdir()),
    )
    engine._tmp_path = tmp_path  # keep reference for cleanup
    engine.ollama = MagicMock()
    engine.ollama.chat.return_value = patch_response
    return engine


def _cleanup_engine(engine: PatchEngine) -> None:
    engine._tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _deep_copy
# ---------------------------------------------------------------------------


def test_deep_copy_dict():
    original = {"a": [1, 2, 3], "b": {"c": True}}
    copy = _deep_copy(original)
    copy["a"].append(99)
    assert original["a"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_replace_operation_success():
    engine = _make_engine(SAMPLE_CONFIG, VALID_PATCH)
    try:
        result = engine.run("Set Alpha value to 99")
        assert result.success, result.errors
        assert result.patch == VALID_PATCH
        assert result.patched_config["scenarios"][0]["value"] == 99
    finally:
        _cleanup_engine(engine)


def test_remove_operation_success():
    engine = _make_engine(SAMPLE_CONFIG, REMOVE_PATCH)
    try:
        result = engine.run("Remove Beta scenario")
        assert result.success, result.errors
        assert len(result.patched_config["scenarios"]) == 1
        assert result.patched_config["scenarios"][0]["id"] == "s1"
    finally:
        _cleanup_engine(engine)


def test_add_operation_success():
    engine = _make_engine(SAMPLE_CONFIG, ADD_PATCH)
    try:
        result = engine.run("Add Gamma scenario")
        assert result.success, result.errors
        assert len(result.patched_config["scenarios"]) == 3
        assert result.patched_config["scenarios"][2]["id"] == "s3"
    finally:
        _cleanup_engine(engine)


def test_diff_snippet_populated_on_success():
    engine = _make_engine(SAMPLE_CONFIG, VALID_PATCH)
    try:
        result = engine.run("Set Alpha value to 99")
        assert result.success
        assert isinstance(result.diff_snippet, str)
    finally:
        _cleanup_engine(engine)


# ---------------------------------------------------------------------------
# Failure / retry paths
# ---------------------------------------------------------------------------


def test_invalid_patch_schema_fails():
    bad_patch = [{"op": "copy", "path": "/foo"}]  # "copy" not in enum
    engine = _make_engine(SAMPLE_CONFIG, bad_patch)
    try:
        result = engine.run("Some instruction")
        assert not result.success
        assert any("validation" in e.lower() or "schema" in e.lower() for e in result.errors)
    finally:
        _cleanup_engine(engine)


def test_bad_json_pointer_fails():
    bad_patch = [{"op": "replace", "path": "/nonexistent/deep/path/999", "value": 1}]
    engine = _make_engine(SAMPLE_CONFIG, bad_patch)
    try:
        result = engine.run("Replace something that doesn't exist")
        assert not result.success
    finally:
        _cleanup_engine(engine)


def test_retry_on_first_failure():
    """Model returns invalid patch first, valid patch on retry."""
    bad_patch = [{"op": "copy", "path": "/foo"}]  # invalid
    good_patch = [{"op": "replace", "path": "/scenarios/0/value", "value": 55}]

    engine = _make_engine(SAMPLE_CONFIG, bad_patch)
    # First call returns bad_patch, second call returns good_patch
    engine.ollama.chat.side_effect = [bad_patch, good_patch]
    try:
        result = engine.run("Set something")
        assert result.success, result.errors
        assert result.patched_config["scenarios"][0]["value"] == 55
    finally:
        _cleanup_engine(engine)


def test_ollama_network_error():
    from ollama_client import OllamaError

    engine = _make_engine(SAMPLE_CONFIG, [])
    engine.ollama.chat.side_effect = OllamaError("connection refused")
    try:
        result = engine.run("Do something")
        assert not result.success
        assert any("Ollama" in e or "failed" in e.lower() for e in result.errors)
    finally:
        _cleanup_engine(engine)


# ---------------------------------------------------------------------------
# active_skill_names
# ---------------------------------------------------------------------------


def test_active_skill_names_empty_project():
    """No SKILL.md in temp dir → empty or global-only skill list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", dir=tmpdir, delete=False
        ) as f:
            json.dump(SAMPLE_CONFIG, f)
            tmp_path = Path(f.name)

        engine = PatchEngine(
            config_paths=[tmp_path],
            project_dir=Path(tmpdir),
        )
        names = engine.active_skill_names()
        assert isinstance(names, list)


def test_validator_rejection():
    """A skill validator that returns errors causes the engine to return failure."""
    engine = _make_engine(SAMPLE_CONFIG, VALID_PATCH)

    # Inject a validator that always rejects
    from skills import SkillFile

    def always_fail(config):
        return ["invariant violated: test error"]

    engine.skills = [SkillFile(path=Path("test.md"), validators=[always_fail])]

    try:
        result = engine.run("Set Alpha value to 99")
        assert not result.success
        assert any("invariant violated" in e for e in result.errors)
    finally:
        _cleanup_engine(engine)


def test_skill_names_filter():
    """skill_names filter includes/excludes the local SKILL.md by stem name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)

        # Place a local SKILL.md in the project dir (stem = "SKILL")
        (project_dir / "SKILL.md").write_text(
            "## Instructions\nLocal skill instructions.\n", encoding="utf-8"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", dir=tmpdir, delete=False
        ) as f:
            json.dump(SAMPLE_CONFIG, f)
            tmp_path = Path(f.name)

        # With matching filter — skill is active
        engine_with = PatchEngine(
            config_paths=[tmp_path],
            project_dir=project_dir,
            skill_names=["SKILL"],
        )
        assert "SKILL" in engine_with.active_skill_names()

        # With non-matching filter — skill is excluded
        engine_without = PatchEngine(
            config_paths=[tmp_path],
            project_dir=project_dir,
            skill_names=["nonexistent"],
        )
        assert engine_without.active_skill_names() == []
