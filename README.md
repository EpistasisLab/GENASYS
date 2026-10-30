# GENASYS

**Genetic Evolution of Networked Agent SYStems**

GENASYS evolves teams of LLM agents that engineer features for a clinical prediction
task. Each team reads a feature catalog built from the cohort, consults literature
(RAG), a knowledge graph and statistical tools, and commits a feature set. That set is
scored with cross-validated XGBoost, and NSGA-II evolves the teams on two objectives at
once: **predictive performance** (ROC-AUC, maximized) and **cost** (USD or tokens,
minimized). The result is a Pareto front of teams, from cheap and adequate to expensive
and strong, each scored once on a held-out test set the agents never see.

Models run either on local Ollama servers or on Azure.

---

## Contents

- [GENASYS](#genasys)
  - [Contents](#contents)
  - [How it works](#how-it-works)
  - [Repository layout](#repository-layout)
  - [Installation](#installation)
  - [Configuration](#configuration)
    - [Model pool: `model_config.json`](#model-pool-model_configjson)
  - [Running](#running)
    - [1. A single run](#1-a-single-run)
  - [Outputs](#outputs)
  - [How a team run can end](#how-a-team-run-can-end)
  - [License](#license)

---

## How it works

**A team** is a Supervisor plus zero or more workers, each an agent with its own model
and sampling parameters (its *blueprint*):

| Role          | What it contributes                                                            |
| ------------- | ------------------------------------------------------------------------------ |
| Supervisor    | Plans, delegates, accepts or rejects candidate features, and finalizes the set |
| Single agent  | Takes the Supervisor's place when a team has no workers                        |
| RAG search    | Grounds candidates in the curated literature store                             |
| KG search     | Walks the knowledge graph for related clinical concepts                        |
| Statistician  | Checks distributions and associations in the dev split                         |
| Domain expert | Reviews candidates for clinical plausibility                                   |

**One evaluation** runs the team's LangGraph graph until the Supervisor calls
`return_final_feature_set`. The committed features are encoded and scored by stratified
5-fold cross-validation within the dev split. Token usage is priced per model and total
token usage (input, output, cached) is recorded. Each team is evaluated `--eval_cnt` times;
its median ROC-AUC and its cost are its two objectives.

**Evolution** follows NSGA-II. Offspring come from crossover or mutation: adding or
removing a teammate, giving an agent a new blueprint, or redrawing one of its evolvable
parameters (for example temperature). A team carries forward only if its median ROC-AUC
is above 0, so at least half of its evaluations must finalize. Survivors are chosen by
non-dominated rank and crowding distance.

**The holdout** (20% by default, split with its own seed) is used once, after the last
generation, to score the final Pareto front.

Evaluations run in parallel as Ray tasks. A deeper walkthrough of every package is in
[Source/README.md](Source/README.md), and each package folder has its own README.

---

## Repository layout

```
GENASYS/
├── Source/                   The framework (see Source/README.md)
│   ├── Agents/  Evaluation/  Factories/  Knowledge/  Models/
│   └── Optimizer/  Reporting/  Task/  Team_Builder/  Utils/
├── .env.example              Example script on how to incorporate API keys
├── .gitignore                Files and folders to ignore
├── requirements.txt          Python packages needed to run
├── LICENSE

```

---

## Installation

Python 3.11 or 3.12.

```bash
git clone https://github.com/EpistasisLab/GENASYS.git
cd GENASYS
```

To install requried packages:

```bash
conda create -n genasys python=3.12 -y
conda activate genasys
pip install -r requirements.txt
```

`requirements.txt` pins `torch`, `transformers` and `tokenizers` together for the
semantic matcher's embeddings; read its comments before changing any of them.

---

## Configuration

### Model pool: `model_config.json`

Every model listed under `models` is in the gene pool. Per model:

| Key                                       | Meaning                                                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `provider`                                | `ollama`, `azure_openai`, `azure_ai` or `azure_anthropic`                                           |
| `endpoint`                                | Server URL (for Ollama, `http://host:port`)                                                         |
| `input_per_million`, `output_per_million` | Price per million tokens                                                                            |
| `param_space`                             | Each knob's allowed values; knobs with several values evolve, single-value knobs are fixed settings |

For local models, the price equals the **active** parameter count in billions (mixture-of-experts
models are priced by their active parameters), so the cost objective tracks compute rather than a vendor price list.

Run-wide settings, all required except the last:

| Key                                            | Meaning                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| `recursion_limit`                              | Graph steps a team may take before it is cut off                       |
| `request_timeout_s`, `default_max_retries`     | Per-request timeout and retries                                        |
| `team_retry_attempts`, `team_retry_delay_secs` | Re-runs of a team after a provider failure                             |
| `abort_after_consecutive_infra_failures`       | Provider-failed teams in a row before the run aborts                   |
| `team_time_limit_s`                            | Optional wall-clock cap per team run; the run then counts as truncated |


## Running

### 1. A single run

```bash
python runner.py \
    --seed 0 --pop_size 10 --gens 4 --eval_cnt 5 --cores 6 \
    --mut_parameters_cnt 1 --prob_mut_topology 0.5 --crossover_prob 0.5 \
    --prob_mut_blueprint 0.2 --prob_mut_parameters 0.5 \
    --task_name task --target_column y \
    --csv_path path_to/data.csv \
    --prompt_dir Prompts/ \
    --catalog_path Feature_Catalog/catalog.json \
    --model_config model_config.json \
    --save_directory Save_Dir/
```

`python runner.py --help` documents every option, including `--holdout_fraction`,
`--split_seed`, `--cv_folds`, `--objective_2` and `--check_models`.

---

## Outputs

Each run writes to `<save_directory>/runs/<task_name>/`:

| File                                               | Contents                                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `archive.json`                                     | Every evaluated team: genome, all evaluations, objectives                                  |
| `team_evaluations.csv`                             | One row per evaluation                                                                     |
| `generation_summary.csv`, `generation_members.csv` | Per-generation statistics and membership                                                   |
| `holdout_results.json`                             | The final Pareto front scored on the holdout                                               |
| `data_split.json`                                  | The dev/holdout split manifest                                                             |
| `Team_Results/Team_<id>/Eval_<i>/`                 | `team.txt`, `run_history.log` (full transcript), per-worker transcripts, feature decisions |
| `Visuals/`                                         | Pareto front plots per generation and GIFs of the run                                      |

SLURM logs go to `Logs/name_<jobid>_<task>.out` and `.err`. Ray can stop forwarding worker
output to the log during a long run; file timestamps under `Team_Results/` are the
reliable sign of progress.

---

## How a team run can end

| Status      | Meaning                                                          | Scored |
| ----------- | ---------------------------------------------------------------- | ------ |
| `finalized` | `return_final_feature_set` was called                            | yes    |
| `stopped`   | Ended without finalizing, for example an empty or text-only turn | 0      |
| `truncated` | Hit `recursion_limit` or `team_time_limit_s`                     | 0      |
| `error`     | The graph raised, including a malformed tool call from the model | 0      |
| `provider`  | The server failed (timeout, connection, 5xx) on every retry      | 0      |

Only `provider` failures are retried, and only they count toward
`abort_after_consecutive_infra_failures`; a model's own mistakes are part of its result.
Full tracebacks are in each evaluation's `run_history.log`.

---

## License

MIT. Copyright (c) 2026 Epistasis Lab at Cedars-Sinai. See [LICENSE](LICENSE).
