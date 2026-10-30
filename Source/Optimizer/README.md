# Optimizer/

The NSGA-II loop that evolves teams. Top of the dependency graph.

| Module | Contents |
|---|---|
| `evolver.py` | `EA`: owns the run (config, task, catalog, user services, factories, topology, archive, `RayBackend`, `EvaluationRecorder`). Runs the generation loop and the holdout stage. Schedules each team's evaluation attempts across Ray workers, pinning each to an Azure resource. |
| `backend.py` | `RayBackend` submits attempts to Ray workers. `InlineBackend` runs them in the calling process, for tests. Both offer `submit`, `wait` and `cancel`. |
| `worker.py` | What a Ray worker runs: `RunContext` (sent to each worker once), `run_attempt`, the Ray task `evaluate_remote`, and the per-process tool cache (`build_tools`, `store_dir`). Each worker builds its matcher, retriever and KG once per run, from the recipes. |
| `preflight.py` | Startup checks: `check_factories_cover_topology`, `factories_in_use`, `probe_matcher` / `probe_retriever` / `probe_kg`, and the opt-in live `check_models`. |
| `reproduction.py` | `Reproduction`: offspring by mutation and crossover within the topology and each `param_space`. |
| `nsga_tool.py` | `non_dominated_sorting`, `crowding_distance`, `nds_tournament`, `nds_truncate`, `front_indices`. |
| `archive.py` | `Archive` (every evaluated team, `reserve` / `finalize`, `archive.json`), `SeenArchive`. |
| `objectives.py` | Objective names (`OBJECTIVE_TOKEN_COUNT`, `OBJECTIVE_COST_USD`), labels, `format_objective_2`. |

`evolver.py` is the only caller of `preflight`, `reproduction`, `nsga_tool`, `archive`,
`backend` and `worker`.

## Used by

| Name | File | Use |
|---|---|---|
| `EA` | `runner.py` | Builds and runs it; `--check_models` calls `EA.check_models`. |
| Objective names | `runner.py` | Choices for `--objective_2`. |
| Objective names, labels, `format_objective_2` | `Reporting/` (all modules except `team_logs.py`, which takes only `PERFORMANCE_LABEL`) | Column names, axis labels, number formatting. |

`objectives.py` imports nothing, so `Reporting/` and `runner.py` can use it without
importing the EA.

## Depends on

Every other package.
