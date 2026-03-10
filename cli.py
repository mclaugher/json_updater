"""
cli.py — Command-line interface for the JSON Patch Editor Agent.

Usage:
    python cli.py edit [--project-dir DIR] [--model MODEL] [--host HOST]
                       [--skill-name NAME [--skill-name NAME ...]]
    python cli.py skills [--project-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from patch_engine import PatchEngine
from schema_infer import load_and_infer, summarize_schema
from skills import discover_skills

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_json_configs(project_dir: Path) -> list[Path]:
    """Return all *.json files directly inside *project_dir* (non-recursive)."""
    configs = sorted(project_dir.glob("*.json"))
    if not configs:
        print(f"[warn] No JSON files found in {project_dir}")
    return configs


def _print_header(label: str, width: int = 60) -> None:
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


def _print_schema_summary(config_paths: list[Path]) -> None:
    """Infer and print the schema summary for the loaded configs."""
    if not config_paths:
        return
    try:
        schema = load_and_infer(config_paths)
        summary = summarize_schema(schema)
        _print_header("Inferred schema summary")
        print(summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Schema inference failed: {exc}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_skills(args: argparse.Namespace) -> None:
    """List active skill files for the given project directory."""
    project_dir = Path(args.project_dir).resolve()
    skills = discover_skills(project_dir)

    if not skills:
        print("No skills active.")
        return

    print(f"Active skills ({len(skills)}):")
    for skill in skills:
        instr_preview = skill.system_instructions[:100].replace("\n", " ")
        print(
            f"\n  [{skill.path.name}]  "
            f"examples={len(skill.examples)}  "
            f"validators={len(skill.validators)}"
        )
        if instr_preview:
            print(f"  instructions: {instr_preview}{'…' if len(skill.system_instructions) > 100 else ''}")
        for i, ex in enumerate(skill.examples):
            print(f"  example {i + 1}: {ex.get('instruction', '(no instruction key)')[:80]}")


def cmd_edit(args: argparse.Namespace) -> None:
    """Interactive edit loop: prompt for instructions, display patch, confirm write."""
    project_dir = Path(args.project_dir).resolve()
    config_paths = _find_json_configs(project_dir)

    if not config_paths:
        print("No JSON config files found. Exiting.")
        sys.exit(1)

    # Parse optional skill name filter (may be None or a list)
    skill_names: list[str] | None = args.skill_name if args.skill_name else None

    # ── Startup banner ──────────────────────────────────────────────────────
    _print_header("JSON Patch Editor — startup")
    print(f"  Project dir  : {project_dir}")
    print(f"  Config files : {[p.name for p in config_paths]}")
    print(f"  Model        : {args.model}")
    print(f"  Ollama host  : {args.host}")
    if skill_names:
        print(f"  Skill filter : {skill_names}")

    # ── Schema inference ────────────────────────────────────────────────────
    _print_schema_summary(config_paths)

    # ── Build engine ────────────────────────────────────────────────────────
    engine = PatchEngine(
        config_paths=config_paths,
        model=args.model,
        host=args.host,
        project_dir=project_dir,
        skill_names=skill_names,
    )

    # ── Connectivity check ──────────────────────────────────────────────────
    if not engine.ollama.ping():
        print(
            f"\n[warn] Cannot reach Ollama at {args.host}. "
            "Make sure 'ollama serve' is running before entering instructions.\n"
        )

    # ── Active skills ────────────────────────────────────────────────────────
    active = engine.active_skill_names()
    _print_header("Active skills")
    if active:
        for name in active:
            print(f"  • {name}")
    else:
        print("  (none)")
        print(f"  Hint: place SKILL.md in {project_dir}")
        print(f"        or in ~/.config/json-editor-skills/ for global skills")

    print("\nType an instruction, or 'quit' to exit.\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        try:
            instruction = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue

        print("  Running…")
        result = engine.run(instruction)

        if not result.success:
            _print_header("ERROR — patch rejected")
            for err in result.errors:
                print(f"  ✗ {err}")
            print()
            continue

        # ── Show patch ───────────────────────────────────────────────────────
        _print_header("Proposed patch")
        print(json.dumps(result.patch, indent=2))

        # ── Show diff ────────────────────────────────────────────────────────
        _print_header("Before / after")
        print(result.diff_snippet)

        # ── Confirm ──────────────────────────────────────────────────────────
        print()
        try:
            confirm = input("Apply patch? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping.")
            continue

        if confirm == "y":
            out_dir = _write_output_dir(project_dir, config_paths, result, instruction)
            print(f"  ✓ Output written to: {out_dir}")
            print("    (original files are unchanged)")
            # Update the engine's in-memory config so the next instruction
            # builds on the patched state without touching the source files.
            target_name = config_paths[0].name
            engine.configs[target_name] = result.patched_config
        else:
            print("  Patch discarded.")
        print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_output_dir(
    project_dir: Path,
    config_paths: list[Path],
    result,
    instruction: str,
) -> Path:
    """Write patched config(s) and a changes summary to a timestamped output directory.

    Original source files are never modified.  The output directory is:
        ``{project_dir}/output/{YYYY-MM-DDTHH-MM-SS}/``

    Contents:
        - ``{original_filename}`` — patched version of the first config
        - ``changes.md`` — human-readable summary of what changed

    Args:
        project_dir: Root directory of the project.
        config_paths: Source config paths (first one is the patched target).
        result: :class:`ApplyResult` from ``PatchEngine.run()``.
        instruction: The natural-language instruction that produced this result.

    Returns:
        Path to the created output directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = project_dir / "output" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write patched config
    target_name = config_paths[0].name
    patched_path = out_dir / target_name
    with open(patched_path, "w", encoding="utf-8") as fh:
        json.dump(result.patched_config, fh, indent=2)
        fh.write("\n")

    # Write changes.md
    changes_path = out_dir / "changes.md"
    with open(changes_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Changes — {timestamp}\n\n")
        fh.write(f"**Source file:** `{config_paths[0]}`  \n")
        fh.write(f"**Patched file:** `{patched_path}`  \n\n")
        fh.write(f"## Instruction\n\n{instruction}\n\n")
        fh.write("## Patch applied\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(result.patch, indent=2))
        fh.write("\n```\n\n")
        fh.write("## Before / after\n\n")
        fh.write("```\n")
        fh.write(result.diff_snippet or "(no diff available)")
        fh.write("\n```\n")

    return out_dir


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="json-patch-editor",
        description="Offline JSON config editor powered by a local Ollama model.",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Directory containing JSON config files (default: current dir).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── edit ─────────────────────────────────────────────────────────────────
    edit_parser = subparsers.add_parser(
        "edit", help="Interactively edit JSON configs using natural-language instructions."
    )
    edit_parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama model name (default: llama3).",
    )
    edit_parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    edit_parser.add_argument(
        "--skill-name",
        action="append",
        metavar="NAME",
        help=(
            "Activate only the skill whose file stem matches NAME. "
            "Can be repeated to enable multiple skills. "
            "Default: all discovered skills are active."
        ),
    )

    # ── skills ───────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "skills",
        help="List all active skills for the project directory.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "edit":
        cmd_edit(args)
    elif args.command == "skills":
        cmd_skills(args)


if __name__ == "__main__":
    main()
