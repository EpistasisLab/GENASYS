# Team_Builder/

What a team is, which teams are allowed, and how one becomes a runnable graph.

| Module | Contents |
|---|---|
| `team.py` | `Team`: roster (role → `BluePrint`), lineage, evaluations, `median_evaluation`. |
| `topology.py` | `SUPERVISOR`, `TopologyRules` (allowed teams), `build_topology_rules`. |
| `graph.py` | `TeamGraphBuilder`, `TeamGraph`, and the `GraphBuilder` / `ModelBuilder` types. |

## Used by

| Name | File | Use |
|---|---|---|
| `Team` | `Optimizer/reproduction.py` | Creates offspring. |
| | `Optimizer/evolver.py` | The population. |
| | `Optimizer/archive.py` | Stored by id. |
| | `Evaluation/evaluator.py` | Evaluated, in a Ray worker. |
| | `Optimizer/evolver.py` | Results recorded with `set_evaluation_result`, once its last attempt lands. |
| | `Reporting/team_logs.py` | Roster written to `team.txt`. |
| `SUPERVISOR` | `Optimizer/reproduction.py`, `Evaluation/evaluator.py`, `Evaluation/run_graph.py` | Tells the lead slot apart from workers. |
| `build_topology_rules` | `Optimizer/evolver.py` | Built once from the config. |
| `TopologyRules` | `Optimizer/reproduction.py` | Samples allowed teams. |
| | `Optimizer/preflight.py` | Checks the factories cover them. |
| `TeamGraphBuilder` | `Optimizer/worker.py` | Built per attempt in each Ray worker, with that attempt's resource; passed to the `Evaluator` as `build_graph`. |
| | `Optimizer/evolver.py` | Built once in the driver, for `check_models`. |
| `GraphBuilder` | `Evaluation/evaluator.py` | Type of `build_graph`; tests pass a scripted one. |
| `ModelBuilder` | `Optimizer/evolver.py`, `Optimizer/preflight.py`, `Optimizer/worker.py` | Type of `build_model`; tests replace the chat models. |

## Depends on

`Factories/`, `Agents/`, `Models/`, `Knowledge/` (`kg_schema_summary`), `Task/`.
