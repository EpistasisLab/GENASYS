# Agents/

What one agent carries: its genome, its graph state, and its tools.

| Module | Contents |
|---|---|
| `blueprint.py` | `BluePrint(model, parameters)`, one agent's genome. |
| `state.py` | `WorkerState`, `TeamState`, the feature-dict reducers, `initial_state`, `pending_entries`, `resolve_feature`. |
| `Tools/feature_state.py` | Tools that stage, review, decide, reject concepts, inspect (`print_*`). |
| `Tools/stats_tools.py` | The four Statistician analyses. |
| `Tools/dataset_tools.py` | `print_dataset_schema`. |
| `Tools/tool_resources.py` | `ToolResources`: catalog, matcher, dev data, retriever, KG. |

## Used by

| Name | File | Use |
|---|---|---|
| `BluePrint` | `Team_Builder/team.py` | Roster values (role → blueprint). |
| | `Models/chat_model.py` | Model and parameters to build. |
| | `Optimizer/reproduction.py` | Mutated and crossed over. |
| | `Optimizer/archive.py` | Serialized into `archive.json`. |
| | `Optimizer/preflight.py`, `Team_Builder/graph.py` | Passed to the model builder. |
| `WorkerState`, `TeamState` | `Team_Builder/graph.py` | State schemas of workers and the Supervisor. |
| `initial_state` | `Evaluation/run_graph.py` | Start state of every run. |
| `pending_entries` | `Evaluation/outcome.py` | Reads the pending queue when a run ends. |
| Tool builders | `Factories/base_factory.py` | Only caller; each factory's flags pick which. |
| `ToolResources` | `Optimizer/evolver.py`, `Optimizer/worker.py` | Built once in the driver, and once per Ray worker process, on the dev split. |
| | `Team_Builder/graph.py` | Passed to every factory's `tools()`. |

## Depends on

`Knowledge/` (catalog, feature encoding), `Task/` (`render_schema_description`).
