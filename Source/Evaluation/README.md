# Evaluation/

Runs a team's graph, scores its features and prices the run.

| Module | Contents |
|---|---|
| `evaluator.py` | `Evaluator.evaluate(team, team_id, eval_index)`: build → run → score → price → log, into the attempt's own `Eval_{i}` folder. Runs in a Ray worker, and retries provider failures there. Holds no state between calls. |
| `evaluator.py` | `EvaluationRecorder` (driver side): `record(result)` appends the run's `team_evaluations.csv` row and counts evaluations lost to the provider in a row. Raises `ProviderAbort` once `abort_after_consecutive_infra_failures` is reached. |
| `run_graph.py` | `invoke_graph` → `GraphRun` (state, error, worker `Handoff`s). `is_infrastructure_error`. Re-raises matcher/retriever/KG contract errors. |
| `outcome.py` | `describe_outcome` → `RunOutcome` (`RunStatus`, `FeatureRecord`s). `feature_decisions`. |
| `scoring.py` | `cross_validate` → `CVScore`; `score_holdout` → `HoldoutScore`; `assert_enough_per_class`; `FOLD_METRICS`. |
| `usage.py` | `aggregate_usage` (tokens per agent), `price_usage` → `RunCost`. |
| `result.py` | `EvalResult` (one evaluation; `team_kwargs()`), `FeatureColumn`, `feature_columns`. |

Inside `evaluator.py`, the steps run in this order: `run_graph` → `outcome` → `scoring` → `usage`, then `result`.

## Errors

| Error | Result |
|---|---|
| Graph error | Team scored 0; full traceback in its `run_history.log`. |
| Provider failure | Retried in the worker. The driver's `EvaluationRecorder` aborts the run after too many in a row. |
| Contract violation or our own bug outside the graph | Crashes the run. |

## Used by

| Name | File | Use |
|---|---|---|
| `Evaluator` | `Optimizer/worker.py` | Runs each attempt in a Ray worker. |
| `EvaluationRecorder`, `ProviderAbort` | `Optimizer/evolver.py` | Records each result on the driver; saves the archive on abort. |
| `assert_enough_per_class` | `Optimizer/evolver.py` | Startup check that the dev split has enough rows for `cv_folds`. |
| `score_holdout` | `Optimizer/evolver.py` | Scores the final front once on the holdout. |
| `HoldoutScore` | `Reporting/holdout.py` | Written to `holdout_results.json`. |
| `EvalResult`, `FeatureColumn`, `RunOutcome`, `GraphRun`, `Handoff`, `FOLD_METRICS` | `Reporting/team_logs.py` | Written to each evaluation's log files. |

## Depends on

`Team_Builder/`, `Agents/` (`initial_state`, `pending_entries`), `Knowledge/`, `Models/`
(`ModelConfig`), `Task/`, `Reporting/team_logs.py`. `team_logs.py` never imports `evaluator.py`,
so no module imports itself back.
