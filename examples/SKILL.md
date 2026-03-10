# Weather Simulation Skill

## Instructions

This project uses a weather simulation config. Vocabulary guide:
- "scenario" refers to an entry in the top-level `scenarios` array.
- "sensor" refers to an entry in the top-level `sensors` array.
- Each scenario has a unique `id` (format: `scenario_NN`), a `name`, a `weather` string,
  `temperature` (float, Celsius), `wind_speed` (integer, km/h), and a `metadata` object.
- When the analyst says "enable" or "disable" a scenario, set `metadata.enabled` to true/false.
- New scenario IDs must follow the pattern `scenario_NN` (zero-padded two-digit integer).
- `priority` values must be unique across all scenarios.

## Examples

```json
{
  "instruction": "Disable the Winter Storm scenario",
  "excerpt": {
    "scenarios": [
      {"id": "scenario_02", "name": "Winter Storm", "weather": "blizzard",
       "metadata": {"enabled": true, "priority": 2}}
    ]
  },
  "patch": [
    {"op": "replace", "path": "/scenarios/1/metadata/enabled", "value": false}
  ]
}
```

```json
{
  "instruction": "Set the temperature of scenario_01 to 32.0",
  "excerpt": {
    "scenarios": [
      {"id": "scenario_01", "name": "Summer Clear", "temperature": 28.5}
    ]
  },
  "patch": [
    {"op": "replace", "path": "/scenarios/0/temperature", "value": 32.0}
  ]
}
```

## Validators

```python
def validate(config: dict) -> list[str]:
    """Ensure all scenario IDs are unique."""
    errors = []
    scenarios = config.get("scenarios", [])
    ids = [s.get("id") for s in scenarios if isinstance(s, dict)]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate scenario IDs detected.")
    return errors
```
