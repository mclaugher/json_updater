# JSON Patch Editor

An offline CLI tool that takes natural-language instructions and produces
validated [RFC 6902 JSON Patch](https://www.rfc-editor.org/rfc/rfc6902) edits
for large JSON config files.  All inference runs locally via
[Ollama](https://ollama.com) — no cloud APIs.

---

## Quick start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# macOS / Linux — see https://ollama.com/download
ollama pull llama3        # recommended default; ~4 GB
ollama serve              # keep this running in a separate terminal
```

**Recommended models** (must support structured outputs / JSON mode):
- `llama3` — good balance of speed and quality, default
- `mistral` — fast, works well for simple edits
- `llama3.1:8b` — more capable for complex patches
- `qwen2.5-coder` — strong for structured/JSON tasks

### 3. Run the editor

```bash
# Edit JSON configs in the examples/ directory
python cli.py --project-dir examples/ edit

# Use a different model
python cli.py --project-dir examples/ edit --model mistral

# Activate only a specific skill by name (file stem, no .md)
python cli.py --project-dir examples/ edit --skill-name SKILL

# List active skills for a project
python cli.py --project-dir examples/ skills
```

Example session:

```
>>> Set the temperature of scenario_01 to 32.0
  Running…

──────────────────────────────────────────────────────────
  Proposed patch
──────────────────────────────────────────────────────────
[
  {
    "op": "replace",
    "path": "/scenarios/0/temperature",
    "value": 32.0
  }
]

──────────────────────────────────────────────────────────
  Before / after
──────────────────────────────────────────────────────────
  REPLACE   /scenarios/0/temperature
             before: 28.5
             after:  32.0

Apply patch? [y/N] y
  ✓ Output written to: examples/output/2026-03-10T14-22-05/
    (original files are unchanged)
```

The original JSON files are **never modified**.  Patched output goes to
`{project-dir}/output/{timestamp}/` along with a `changes.md` summary.

---

## How it works

1. **Schema inference** — on startup, all `*.json` files in the project directory
   are scanned and a JSON Schema is inferred automatically.
2. **Context extraction** — only the relevant portion of the config is sent to
   the model (keyword-matched subtrees, capped at ~50 lines).
3. **Patch generation** — Ollama is called with your instruction and a structured
   output schema that forces it to emit a JSON Patch array.
4. **Validation** — the patch is validated against the RFC 6902 schema, applied
   to an in-memory copy, and then re-validated against the inferred schema.
5. **Review & confirm** — you see the patch and a before/after diff before anything
   is written.

---

## Authoring a SKILL.md

A skill file teaches the editor about your domain: vocabulary, invariants, and
example patches.  Place it at **`{project-dir}/SKILL.md`** (project-local) or in
**`~/.config/json-editor-skills/`** (global, applied to all projects).

A skill file is a Markdown document with up to three `##` sections:

```markdown
# My Domain Skill

## Instructions

Plain text appended to the system prompt.  Explain domain vocabulary,
naming conventions, and rules that cannot be inferred from the schema.

Example:
- "scenario" means an entry in the top-level `scenarios` array.
- Scenario IDs follow the format `scenario_NN` (zero-padded two digits).
- When the analyst says "enable", set `metadata.enabled` to true.

## Examples

One or more fenced ```json blocks.  Each block is an object with three keys:
- "instruction" — a sample natural-language instruction
- "excerpt"     — the relevant JSON subtree
- "patch"       — the correct patch array

\`\`\`json
{
  "instruction": "Disable the Winter Storm scenario",
  "excerpt": {
    "scenarios": [
      {"id": "scenario_02", "name": "Winter Storm",
       "metadata": {"enabled": true}}
    ]
  },
  "patch": [
    {"op": "replace", "path": "/scenarios/1/metadata/enabled", "value": false}
  ]
}
\`\`\`

## Validators

One or more fenced ```python blocks.  Each block **must** define exactly one
function named `validate(config: dict) -> list[str]`.  Return an empty list
for a valid config, or a list of error strings to reject the patch.

\`\`\`python
def validate(config: dict) -> list[str]:
    """Ensure all scenario IDs are unique."""
    errors = []
    ids = [s.get("id") for s in config.get("scenarios", []) if isinstance(s, dict)]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate scenario IDs detected.")
    return errors
\`\`\`
```

### Security note

Validator blocks are executed as Python code using `exec()`.  Only load
SKILL.md files from sources you trust — treat them the same as any other
code you run on your machine.

### Debugging your SKILL.md

Parsing errors (bad JSON in `## Examples`, syntax errors in `## Validators`)
are printed to **stderr** with a `[SKILL.md error]` prefix and the file path.
Check stderr output when the active-skills list shows fewer skills than expected.

---

## CLI reference

```
python cli.py --project-dir DIR <subcommand> [options]

Global options:
  --project-dir DIR   Directory containing *.json config files (default: .)

Subcommands:
  edit                Interactive edit loop
    --model MODEL     Ollama model name (default: llama3)
    --host  URL       Ollama server URL (default: http://localhost:11434)
    --skill-name NAME Activate only this skill (by file stem, repeatable)

  skills              List active skills for the project directory
```

---

## Output directory

After confirming a patch, output is written to:

```
{project-dir}/output/{YYYY-MM-DDTHH-MM-SS}/
  {original-filename}    ← patched JSON
  changes.md             ← instruction, patch array, before/after diff
```

To apply the result, copy the patched file to your desired location.
Chained edits within the same session build on each other in memory.

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests use a stubbed Ollama client — no running server required.
