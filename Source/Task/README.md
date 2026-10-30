# Task/

The dataset, the target and the dev/holdout split.

| Module | Contents |
|---|---|
| `task.py` | `Task`: loads the CSV, checks the target is a clean 0/1 column with both classes, holds dev and holdout. `render_schema_description`. |
| `data_split.py` | `split_dev_holdout` (stratified), `describe_split`, `write_split_manifest`. |

## Used by

| Name | File | Use |
|---|---|---|
| `Task` | `Optimizer/evolver.py` | Built once per run; the holdout is read only at the end. |
| | `Evaluation/evaluator.py` | Dev data and target column for scoring. |
| | `Team_Builder/graph.py` | Target and file name in prompts. |
| `render_schema_description` | `Agents/Tools/dataset_tools.py` | `print_dataset_schema` output. |
| `describe_split`, `write_split_manifest` | `Optimizer/evolver.py` | Console line and split manifest. |
| `split_dev_holdout` | `task.py` | Internal. |

## Depends on

Nothing in `Source/`.
