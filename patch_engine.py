"""
patch_engine.py — Orchestrates the generate → validate → apply → revalidate loop.

The engine is intentionally domain-agnostic. It operates entirely on in-memory
copies of the loaded configs; callers decide whether to write results to disk.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonpatch
import jsonschema
from jsonpointer import JsonPointerException, resolve_pointer

from context_extractor import extract_context
from ollama_client import OllamaClient
from patch_schema import PATCH_ARRAY_SCHEMA, validate_patch
from schema_infer import load_and_infer, summarize_schema
from skills import collect_validators, discover_skills, merge_skill_instructions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — generic JSON Patch rules, injected once per session.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are a JSON configuration editor.
Your ONLY output is a raw JSON array of patch operations (RFC 6902 subset).

═══ PATCH OPERATION RULES ═══════════════════════════════════════════════════
Each element of the array must be an object with:
  "op"    — one of: "add", "remove", "replace"
  "path"  — a JSON Pointer (RFC 6901), e.g. "/scenarios/0/weather"
  "value" — required for "add" and "replace"; must be omitted for "remove"

JSON Pointer rules:
  • Use integer indices for array elements: /array/0, /array/1, …
  • Use "-" to append to an array: /array/-
  • Escape "/" in key names as "~1"; escape "~" as "~0"

═══ OUTPUT RULES ════════════════════════════════════════════════════════════
  • Output ONLY the raw JSON array — no explanation, no prose, no fences.
  • Use the EXACT paths visible in the JSON excerpt below.
  • Do NOT invent paths that are not present in the excerpt.
  • Do NOT output the full config — only the minimal patch needed.
  • If the instruction cannot be satisfied safely, output an empty array: []
"""

# ---------------------------------------------------------------------------
# User message templates — each slot is clearly delimited.
# ---------------------------------------------------------------------------

_USER_TMPL = """\
=== INSTRUCTION ===
{instruction}

=== CONTEXT SUMMARY ===
{context_summary}

=== JSON EXCERPT ===
(Use the paths visible here. Array indices start at 0.)
{json_excerpt}

=== YOUR TASK ===
Output the JSON patch array that fulfils the instruction above.\
"""

_RETRY_TMPL = """\
=== INSTRUCTION ===
{instruction}

=== CONTEXT SUMMARY ===
{context_summary}

=== JSON EXCERPT ===
{json_excerpt}

=== ERROR REPORT (previous attempt was rejected) ===
{error_report}

=== YOUR TASK ===
The previous patch was rejected for the reasons above.
Output a corrected JSON patch array that fixes every error listed.\
"""

_MISSING = object()  # sentinel for "path not found"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ApplyResult:
    """Result of a single PatchEngine.run() call.

    Attributes:
        patch: The validated patch array (list of operation dicts).
        patched_config: The in-memory config after applying the patch.
        diff_snippet: Human-readable before/after for each patch operation.
        success: True if patch was applied and all validations passed.
        errors: List of error strings accumulated during the run.
    """

    patch: list[dict]
    patched_config: dict | list
    diff_snippet: str
    success: bool
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PatchEngine:
    """Coordinate the full edit cycle for a set of JSON config files.

    Args:
        config_paths: List of paths to JSON config files to load.
        model: Ollama model name.
        host: Ollama server base URL.
        project_dir: Project root used for skill discovery.
        skill_names: If given, only skills whose file stem matches one of
            these names are activated (case-sensitive).
    """

    def __init__(
        self,
        config_paths: list[Path],
        model: str = "llama3",
        host: str = "http://localhost:11434",
        project_dir: Path | None = None,
        skill_names: list[str] | None = None,
    ) -> None:
        self.config_paths = config_paths
        self.model = model
        self.host = host
        self.project_dir = project_dir or Path(".")

        # Load all configs keyed by filename.
        self.configs: dict[str, Any] = {}
        for p in config_paths:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    self.configs[p.name] = json.load(fh)
            except json.JSONDecodeError as exc:
                print(
                    f"[error] Could not parse {p}: {exc.msg} "
                    f"(line {exc.lineno}, col {exc.colno})",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Infer schema from current config state and cache the summary.
        self.schema = load_and_infer(config_paths) if config_paths else {}
        self.schema_summary: str = summarize_schema(self.schema) if self.schema else ""

        # Discover and optionally filter skills.
        all_skills = discover_skills(self.project_dir)
        if skill_names:
            self.skills = [s for s in all_skills if s.path.stem in skill_names]
            skipped = len(all_skills) - len(self.skills)
            if skipped:
                logger.debug("Filtered out %d skill(s) not in %s", skipped, skill_names)
        else:
            self.skills = all_skills

        logger.debug("Active skills: %s", [s.path.name for s in self.skills])

        self.ollama = OllamaClient(model=model, host=host)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, instruction: str) -> ApplyResult:
        """Execute one edit cycle for *instruction*.

        Does NOT write to disk. The caller receives an :class:`ApplyResult`
        and decides whether to persist the patched config.

        Steps:
          1. extract_context → json_excerpt, context_summary
          2. Build system prompt (generic rules + skill instructions + examples)
          3. Call Ollama with PATCH_ARRAY_SCHEMA as the format constraint
          4. validate_patch (patch schema check)
          5. Apply patch with jsonpatch to an in-memory copy
          6. Validate patched copy against inferred schema + custom validators
          7. On any failure: build error_report, retry once
          8. Return ApplyResult

        Args:
            instruction: Natural-language edit instruction.

        Returns:
            An :class:`ApplyResult`.
        """
        target_name, target_config = next(iter(self.configs.items()))
        working_copy = _deep_copy(target_config)

        # ── Step 1: Context extraction ──────────────────────────────────────
        json_excerpt, context_summary = extract_context(working_copy, instruction)

        # ── Step 2: Prompts ─────────────────────────────────────────────────
        system_prompt = self._build_system_prompt()
        user_message = _USER_TMPL.format(
            instruction=instruction,
            context_summary=context_summary,
            json_excerpt=json.dumps(json_excerpt, indent=2),
        )

        # ── Step 3–4: Generate + validate patch ─────────────────────────────
        patch, errors = self._call_and_validate(system_prompt, user_message)
        if errors:
            logger.info("Attempt 1 failed (%d error(s)); retrying with error report.", len(errors))
            retry_msg = _RETRY_TMPL.format(
                instruction=instruction,
                context_summary=context_summary,
                json_excerpt=json.dumps(json_excerpt, indent=2),
                error_report="\n".join(errors),
            )
            patch, errors = self._call_and_validate(system_prompt, retry_msg)

        if errors:
            return ApplyResult(
                patch=patch,
                patched_config=working_copy,
                diff_snippet="",
                success=False,
                errors=errors,
            )

        # ── Step 5: Apply patch ──────────────────────────────────────────────
        try:
            patched = jsonpatch.apply_patch(working_copy, patch, in_place=False)
        except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as exc:
            return ApplyResult(
                patch=patch,
                patched_config=working_copy,
                diff_snippet="",
                success=False,
                errors=[f"Patch application error: {exc}"],
            )

        # ── Step 6: Validate patched config ─────────────────────────────────
        validation_errors = self._validate_config(patched)
        if validation_errors:
            logger.info("Post-apply validation failed; retrying with error report.")
            retry_msg2 = _RETRY_TMPL.format(
                instruction=instruction,
                context_summary=context_summary,
                json_excerpt=json.dumps(json_excerpt, indent=2),
                error_report="\n".join(validation_errors),
            )
            patch2, gen_errors2 = self._call_and_validate(system_prompt, retry_msg2)
            if not gen_errors2:
                try:
                    patched2 = jsonpatch.apply_patch(working_copy, patch2, in_place=False)
                    val_errors2 = self._validate_config(patched2)
                    if not val_errors2:
                        patch, patched, validation_errors = patch2, patched2, []
                except Exception:
                    pass  # fall through and return the original failure

        if validation_errors:
            return ApplyResult(
                patch=patch,
                patched_config=working_copy,
                diff_snippet="",
                success=False,
                errors=validation_errors,
            )

        # ── Step 7: Build diff and return ────────────────────────────────────
        diff_snippet = _build_diff_snippet(working_copy, patched, patch)

        return ApplyResult(
            patch=patch,
            patched_config=patched,
            diff_snippet=diff_snippet,
            success=True,
        )

    def active_skill_names(self) -> list[str]:
        """Return the stem names of all active skill files."""
        return [s.path.stem for s in self.skills]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt.

        Structure:
          1. Generic JSON Patch rules (_SYSTEM_PROMPT_BASE)
          2. Domain instructions from skills (if any)
          3. Few-shot examples from skills (if any), as plain-text blocks
        """
        parts: list[str] = [_SYSTEM_PROMPT_BASE]

        skill_instructions = merge_skill_instructions(self.skills)
        if skill_instructions:
            parts.append(
                "═══ DOMAIN-SPECIFIC INSTRUCTIONS ═══════════════════════════════════\n"
                + skill_instructions
            )

        for skill in self.skills:
            for ex in skill.examples:
                ex_instruction = ex.get("instruction", "")
                ex_excerpt = json.dumps(ex.get("excerpt", {}), indent=2)
                ex_patch = json.dumps(ex.get("patch", []), indent=2)
                parts.append(
                    "═══ EXAMPLE ═══════════════════════════════════════════════════════\n"
                    f"Instruction: {ex_instruction}\n"
                    f"Excerpt:\n{ex_excerpt}\n"
                    f"Patch:\n{ex_patch}"
                )

        return "\n\n".join(parts)

    def _call_and_validate(
        self,
        system_prompt: str,
        user_message: str,
    ) -> tuple[list[dict], list[str]]:
        """Call Ollama and validate the patch schema of the response.

        Returns:
            ``(patch, errors)`` — *errors* is empty on success.
        """
        try:
            raw = self.ollama.chat(
                system=system_prompt,
                user=user_message,
                format_schema=PATCH_ARRAY_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
            return [], [f"Ollama call failed: {exc}"]

        if not isinstance(raw, list):
            return [], [f"Model returned a {type(raw).__name__}, expected a JSON array."]

        try:
            validate_patch(raw)
        except jsonschema.ValidationError as exc:
            return raw, [f"Patch schema validation failed: {exc.message}"]

        return raw, []

    def _validate_config(self, config: dict | list) -> list[str]:
        """Validate *config* against the inferred schema and skill validators.

        Returns:
            List of error strings; empty list means valid.
        """
        errors: list[str] = []

        if self.schema:
            try:
                jsonschema.validate(instance=config, schema=self.schema)
            except jsonschema.ValidationError as exc:
                errors.append(f"Schema validation: {exc.message}")

        for skill, validator in collect_validators(self.skills):
            try:
                result = validator(config)
                if result:
                    errors.extend(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"Validator in {skill.path.name} raised "
                    f"{type(exc).__name__}: {exc}"
                )

        return errors


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _deep_copy(obj: Any) -> Any:
    """Deep copy via JSON round-trip (safe for all JSON-serialisable data)."""
    return json.loads(json.dumps(obj))


def _build_diff_snippet(
    original: dict | list,
    patched: dict | list,
    patch_ops: list[dict],
) -> str:
    """Build a path-by-path before/after diff from the patch operations.

    Uses ``jsonpointer.resolve_pointer`` to look up the actual value at each
    operation's path in both the original and patched configs, producing a
    concise human-readable summary.

    Args:
        original: Config before the patch.
        patched: Config after the patch.
        patch_ops: The applied patch operations.

    Returns:
        A multi-line string suitable for terminal display.
    """
    if not patch_ops:
        return "  (no operations)"

    lines: list[str] = []
    for op in patch_ops:
        path: str = op["path"]
        op_type: str = op["op"]

        # The "-" token in JSON Pointer means "append to array".
        # resolve_pointer returns an EndOfList sentinel for that path, which
        # is not JSON-serialisable.  Resolve the last element of the parent
        # array instead.
        effective_path = path
        if path.endswith("/-"):
            parent_path = path[:-2] or "/"
            parent_after = resolve_pointer(patched, parent_path, _MISSING)
            if parent_after is not _MISSING and isinstance(parent_after, list) and parent_after:
                effective_path = f"{parent_path}/{len(parent_after) - 1}"
            else:
                effective_path = path  # fallback; _fmt will handle gracefully

        before = resolve_pointer(original, effective_path, _MISSING)
        after = resolve_pointer(patched, effective_path, _MISSING)

        lines.append(f"  {op_type.upper():8s}  {path}")

        if op_type == "remove":
            if before is not _MISSING:
                lines.append(f"             before: {_fmt(before)}")
        elif op_type == "add":
            if after is not _MISSING:
                lines.append(f"             added:  {_fmt(after)}")
        else:  # replace
            if before is not _MISSING:
                lines.append(f"             before: {_fmt(before)}")
            if after is not _MISSING:
                lines.append(f"             after:  {_fmt(after)}")

    return "\n".join(lines)


def _fmt(value: Any, max_len: int = 120) -> str:
    """Compact JSON representation of *value*, truncated to *max_len* chars."""
    try:
        s = json.dumps(value)
    except (TypeError, ValueError):
        s = repr(value)
    return s if len(s) <= max_len else s[:max_len] + "…"
