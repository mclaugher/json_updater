"""
skills.py — Discover, parse, and merge SKILL.md files (global + per-project).

A skill file is a Markdown document with optional sections:
  ## Instructions  — plain text appended to the system prompt
  ## Examples      — fenced JSON blocks with instruction/excerpt/patch triples
  ## Validators    — fenced ```python blocks; each must define
                     ``validate(config: dict) -> list[str]``
"""

from __future__ import annotations

import builtins
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_GLOBAL_SKILLS_DIR = Path.home() / ".config" / "json-editor-skills"
_LOCAL_SKILL_FILE = "SKILL.md"

# Regex to find fenced code blocks: ```[lang]\n...\n```
_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


@dataclass
class SkillFile:
    """Parsed representation of a single SKILL.md file.

    Attributes:
        path: Filesystem path to the source markdown file.
        system_instructions: Text to prepend to the system prompt.
        examples: List of few-shot example dicts with keys
            ``instruction``, ``excerpt``, and ``patch``.
        validators: List of callables ``(config: dict) -> list[str]``
            extracted from ``## Validators`` code blocks.
    """

    path: Path
    system_instructions: str = ""
    examples: list[dict] = field(default_factory=list)
    validators: list[Callable] = field(default_factory=list)


def discover_skills(project_dir: Path) -> list[SkillFile]:
    """Discover and parse all applicable skill files.

    Loads skills from:
    1. ``~/.config/json-editor-skills/*.md`` (global, alphabetical order)
    2. ``{project_dir}/SKILL.md`` (local, loaded last so it takes precedence)

    Args:
        project_dir: The project directory to search for a local SKILL.md.

    Returns:
        List of parsed :class:`SkillFile` objects (may be empty).
    """
    skill_files: list[SkillFile] = []

    # Global skills
    if _GLOBAL_SKILLS_DIR.is_dir():
        for md_path in sorted(_GLOBAL_SKILLS_DIR.glob("*.md")):
            try:
                skill_files.append(parse_skill(md_path))
                logger.debug("Loaded global skill: %s", md_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse global skill %s: %s", md_path, exc)

    # Local skill
    local_path = project_dir / _LOCAL_SKILL_FILE
    if local_path.is_file():
        try:
            skill_files.append(parse_skill(local_path))
            logger.debug("Loaded local skill: %s", local_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse local skill %s: %s", local_path, exc)

    return skill_files


def parse_skill(path: Path) -> SkillFile:
    """Parse a SKILL.md file into a :class:`SkillFile`.

    Recognised sections (case-insensitive ``##`` headings):
    - ``## Instructions`` — text after heading until next ``##`` or EOF
    - ``## Examples`` — fenced JSON blocks parsed as example dicts
    - ``## Validators`` — fenced ``python`` blocks exec'd to extract
      a ``validate(config) -> list[str]`` callable

    Args:
        path: Path to the Markdown skill file.

    Returns:
        A populated :class:`SkillFile`.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    text = path.read_text(encoding="utf-8")
    sections = _split_sections(text)

    instructions = sections.get("instructions", "").strip()
    examples = _parse_examples(sections.get("examples", ""), path)
    validators = _parse_validators(sections.get("validators", ""), path)

    return SkillFile(
        path=path,
        system_instructions=instructions,
        examples=examples,
        validators=validators,
    )


def merge_skill_instructions(skills: list[SkillFile]) -> str:
    """Concatenate system instruction blocks from all skills.

    Args:
        skills: List of :class:`SkillFile` objects.

    Returns:
        A single string with each skill's instructions separated by a blank line.
    """
    parts = [s.system_instructions for s in skills if s.system_instructions.strip()]
    return "\n\n".join(parts)


def collect_validators(skills: list[SkillFile]) -> list[tuple["SkillFile", Callable]]:
    """Return all validators paired with their source skill.

    Returning ``(SkillFile, Callable)`` pairs lets callers include the skill
    filename in error messages when a validator raises at runtime.

    Args:
        skills: List of :class:`SkillFile` objects.

    Returns:
        Flat list of ``(skill, validate_fn)`` tuples.
    """
    pairs: list[tuple[SkillFile, Callable]] = []
    for skill in skills:
        for fn in skill.validators:
            pairs.append((skill, fn))
    return pairs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> dict[str, str]:
    """Split a Markdown string into section bodies keyed by heading name.

    Only ``##`` (level-2) headings are recognised as section delimiters.
    The returned keys are lower-cased heading text (stripped).
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for line in text.splitlines(keepends=True):
        heading_match = re.match(r"^##\s+(.+)", line)
        if heading_match:
            if current_key is not None:
                sections[current_key] = "".join(buffer)
            current_key = heading_match.group(1).strip().lower()
            buffer = []
        else:
            if current_key is not None:
                buffer.append(line)

    if current_key is not None:
        sections[current_key] = "".join(buffer)

    return sections


def _parse_examples(section_text: str, source_path: Path) -> list[dict]:
    """Extract fenced JSON blocks from the Examples section."""
    examples: list[dict] = []
    for i, match in enumerate(_FENCE_RE.finditer(section_text)):
        lang, code = match.group(1), match.group(2).strip()
        if lang in ("json", ""):
            try:
                obj = json.loads(code)
                if isinstance(obj, dict):
                    examples.append(obj)
            except json.JSONDecodeError as exc:
                print(
                    f"[SKILL.md error] Malformed JSON in example block {i} — skipping.\n"
                    f"  File : {source_path}\n"
                    f"  Error: {exc}",
                    file=sys.stderr,
                )
    return examples


def _parse_validators(section_text: str, source_path: Path) -> list[Callable]:
    """Exec fenced Python blocks and extract ``validate`` callables.

    Security note: validators are trusted local config, like any project file.
    Only ``builtins`` are available in the exec namespace.

    Args:
        section_text: Text of the ``## Validators`` section.
        source_path: Used for error messages only.

    Returns:
        List of callables named ``validate``.
    """
    validators: list[Callable] = []
    for i, match in enumerate(_FENCE_RE.finditer(section_text)):
        lang, code = match.group(1), match.group(2).strip()
        if lang not in ("python", "py"):
            continue
        namespace: dict = {"__builtins__": builtins}
        try:
            exec(compile(code, f"{source_path}:block{i}", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            print(
                f"[SKILL.md error] Failed to compile validator block {i} — skipping.\n"
                f"  File : {source_path}\n"
                f"  Error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        validate_fn = namespace.get("validate")
        if callable(validate_fn):
            validators.append(validate_fn)
        else:
            print(
                f"[SKILL.md error] Validator block {i} does not define a callable named 'validate' — skipping.\n"
                f"  File : {source_path}\n"
                f"  Hint : Each ```python block must define: def validate(config: dict) -> list[str]",
                file=sys.stderr,
            )
    return validators
