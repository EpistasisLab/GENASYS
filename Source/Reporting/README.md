# Reporting/

Writes results. Computes nothing.

| Module | Writes |
|---|---|
| `team_logs.py` | `Team_Results/Team_{id}/Eval_{i}/`: `team.txt`, `run_history.log`, `{role}_transcripts.json`, `feature_decisions.json`, `feature_mean_details.csv`, `feature_fold{k}_details.csv`, `Provider_Failures/`. `team_evaluations.csv` is written by `EvaluationRecorder` through `append_team_evaluation`. |
| `generation_summary.py` | Console summary, `generation_summary.csv`, `generation_members.csv`. |
| `plots.py` | `plot_front`: one generation's Pareto front PNG. |
| `holdout.py` | `holdout_results.json` and the final console summary. |
| `run_history.py` | Reads a run folder back into a `RunHistory`. |
| `animations.py` | Per-generation frames and GIFs under `{run_dir}/Visuals/`. |

## Used by

| File | Uses |
|---|---|
| `Evaluation/evaluator.py` | `team_logs`: every `write_*`, `print_evaluation`, `eval_dir`, `attempt_dir`, and `append_team_evaluation` (via `EvaluationRecorder`). |
| `Optimizer/evolver.py` | Each generation: `generation_summary`, `plot_front`, `save_generation_frames`. After the run: `holdout`. |
| `runner.py` | `make_run_gifs` after the run. |
| `animations.py` | `run_history` (internal). |

## Depends on

`Evaluation/` result types (`team_logs.py`, `holdout.py`), `Optimizer/objectives.py`,
`Team_Builder/` (`Team`), `Utils/` (`format_duration`).
