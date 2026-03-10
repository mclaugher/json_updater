"""Tests for skills.py."""

import tempfile
from pathlib import Path

import pytest

from skills import (
    SkillFile,
    collect_validators,
    discover_skills,
    merge_skill_instructions,
    parse_skill,
)


SKILL_MD_CONTENT = """\
# Test Skill

## Instructions

Always use integer IDs.
Keep scenario names short.

## Examples

```json
{
  "instruction": "Set temperature to 20",
  "excerpt": {"scenarios": [{"id": "s1", "temperature": 15}]},
  "patch": [{"op": "replace", "path": "/scenarios/0/temperature", "value": 20}]
}
```

## Validators

```python
def validate(config: dict) -> list:
    errors = []
    if not isinstance(config.get("scenarios"), list):
        errors.append("scenarios must be a list")
    return errors
```
"""


# ---------------------------------------------------------------------------
# parse_skill
# ---------------------------------------------------------------------------


def _write_skill(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return Path(f.name)


def test_parse_skill_instructions():
    path = _write_skill(SKILL_MD_CONTENT)
    try:
        skill = parse_skill(path)
        assert "integer IDs" in skill.system_instructions
        assert "scenario names short" in skill.system_instructions
    finally:
        path.unlink()


def test_parse_skill_examples():
    path = _write_skill(SKILL_MD_CONTENT)
    try:
        skill = parse_skill(path)
        assert len(skill.examples) == 1
        ex = skill.examples[0]
        assert ex["instruction"] == "Set temperature to 20"
        assert ex["patch"][0]["op"] == "replace"
    finally:
        path.unlink()


def test_parse_skill_validators():
    path = _write_skill(SKILL_MD_CONTENT)
    try:
        skill = parse_skill(path)
        assert len(skill.validators) == 1
        fn = skill.validators[0]
        assert callable(fn)

        # Valid config
        errors = fn({"scenarios": []})
        assert errors == []

        # Invalid config
        errors = fn({"scenarios": "not-a-list"})
        assert len(errors) == 1
    finally:
        path.unlink()


def test_parse_skill_empty_file():
    path = _write_skill("")
    try:
        skill = parse_skill(path)
        assert skill.system_instructions == ""
        assert skill.examples == []
        assert skill.validators == []
    finally:
        path.unlink()


def test_parse_skill_no_validators_section():
    content = "## Instructions\n\nJust instructions here.\n"
    path = _write_skill(content)
    try:
        skill = parse_skill(path)
        assert "Just instructions" in skill.system_instructions
        assert skill.validators == []
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# merge_skill_instructions
# ---------------------------------------------------------------------------


def test_merge_skill_instructions_single():
    skill = SkillFile(path=Path("x.md"), system_instructions="Do thing A.")
    result = merge_skill_instructions([skill])
    assert result == "Do thing A."


def test_merge_skill_instructions_multiple():
    s1 = SkillFile(path=Path("a.md"), system_instructions="Rule 1.")
    s2 = SkillFile(path=Path("b.md"), system_instructions="Rule 2.")
    result = merge_skill_instructions([s1, s2])
    assert "Rule 1." in result
    assert "Rule 2." in result


def test_merge_skill_instructions_empty_skipped():
    s1 = SkillFile(path=Path("a.md"), system_instructions="")
    s2 = SkillFile(path=Path("b.md"), system_instructions="Has content.")
    result = merge_skill_instructions([s1, s2])
    assert result == "Has content."


# ---------------------------------------------------------------------------
# collect_validators
# ---------------------------------------------------------------------------


def test_collect_validators():
    def v1(c):
        return []

    def v2(c):
        return ["err"]

    s1 = SkillFile(path=Path("a.md"), validators=[v1])
    s2 = SkillFile(path=Path("b.md"), validators=[v2])
    pairs = collect_validators([s1, s2])
    assert len(pairs) == 2
    # Each element is a (SkillFile, callable) tuple
    skill0, fn0 = pairs[0]
    skill1, fn1 = pairs[1]
    assert skill0.path == Path("a.md")
    assert fn0({}) == []
    assert skill1.path == Path("b.md")
    assert fn1({}) == ["err"]


def test_parse_skill_malformed_json_example(capsys):
    """Malformed JSON in ## Examples → stderr message, examples list is empty."""
    content = "## Examples\n\n```json\n{bad json here\n```\n"
    path = _write_skill(content)
    try:
        skill = parse_skill(path)
        assert skill.examples == []
        captured = capsys.readouterr()
        assert "[SKILL.md error]" in captured.err
        assert str(path) in captured.err
    finally:
        path.unlink()


def test_parse_skill_validator_syntax_error(capsys):
    """Python syntax error in ## Validators → stderr message, validators list is empty."""
    content = "## Validators\n\n```python\ndef validate(config):\n  return [  # missing bracket\n```\n"
    path = _write_skill(content)
    try:
        skill = parse_skill(path)
        assert skill.validators == []
        captured = capsys.readouterr()
        assert "[SKILL.md error]" in captured.err
        assert str(path) in captured.err
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# discover_skills
# ---------------------------------------------------------------------------


def test_discover_skills_local_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        skill_path = project_dir / "SKILL.md"
        skill_path.write_text(SKILL_MD_CONTENT, encoding="utf-8")

        skills = discover_skills(project_dir)
        local_skill_names = [s.path.name for s in skills]
        assert "SKILL.md" in local_skill_names


def test_discover_skills_no_skill_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        # Only global skills would load; since global dir likely absent, result is []
        skills = discover_skills(project_dir)
        # No local skill — list may be empty or contain only global skills
        local = [s for s in skills if s.path.name == "SKILL.md"]
        assert local == []
