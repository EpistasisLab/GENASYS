# Utils/

Shared logging helpers. The bottom layer: it imports nothing from `Source/`.

| Module | Contents |
|---|---|
| `log.py` | `warn(message, source)`: prints `[source] WARNING: …`. `format_duration(seconds)`. |

## Used by

| Name | File |
|---|---|
| `warn` | `Models/model_config.py`, `Team_Builder/topology.py`, `Optimizer/evolver.py`, `Optimizer/preflight.py`, `Optimizer/archive.py` |
| `format_duration` | `Reporting/generation_summary.py` |
