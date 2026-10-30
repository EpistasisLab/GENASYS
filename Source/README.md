# Source/

GENASYS evolves teams of LLM agents that propose features for a prediction task. Each team
is scored on two objectives with NSGA-II: ROC-AUC (maximized) and token count or USD cost
(minimized). `runner.py` at the repo root builds the factories and the EA, then runs it.

## Running

`runner.py` is the generic entry point. It builds nothing and assumes every artifact
exists: the cohort CSV, the prompts (`--prompt_dir`), the feature catalog
(`--catalog_path`), the model config (`--model_config`), the RAG vector store and the
knowledge graph. The semantic matcher, the literature retriever and the knowledge graph
are passed as recipes that `main` builds from a JSON object of constructor arguments
(`MATCHER_CONFIG`, `RETRIEVER_CONFIG`, `KG_CONFIG`); to use others, import your
`SemanticMatcher`, `Retriever` or `KnowledgeGraph` subclass and point the constant at its
config there. Each process that needs them builds its own: the driver once, to probe them,
and each Ray worker once per run. `--rag_store` replaces only the retriever config's
`vector_store_path`. Opening a Chroma store rewrites its `chroma.sqlite3`, so every process
opens a copy of its own, made under the run folder's `worker_stores/`.
`submit.sh` is the SLURM array example: each task copies the master store into its own
save directory (`results/<task>/seed_<id>/rag_store`) and runs with `--seed <id>`. Its entry function is `main(args)`; `build_parser()` gives the arguments.

Each generation's evaluations run in parallel on Ray. `--cores` is the number of worker
processes and the most attempts in flight at once. A team's `--eval_cnt` evaluations are
separate attempts, each its own Ray task. `--eval_cnt` is required and independent of
`--cores`. Launches are at least `--stagger_seconds` apart (default 5) so teams do not all
start their LLM calls at once. When Azure resources are configured (`R{n}_`),
`ResourceAllocator` pins each attempt to one, so no two in-flight attempts share a
(resource, model) deployment. The Ray dashboard starts with the run, and its URL is
printed at startup (port 8265, or the next free port).


## Folders

Each folder has its own `README.md`: its modules, which files use what, and what it depends on.

| Package | Answers |
|---|---|
| `Factories/` | What does each role say, and which tools does it get? |
| `Agents/` | What does one agent carry (blueprint, tools, state)? |
| `Team_Builder/` | How does a `Team` become a runnable graph? |
| `Evaluation/` | How do we run that graph, score the features and price the run? |
| `Knowledge/` | What do agent tools and scoring both read (catalog, feature encoding, semantic matcher, retriever, KG)? |
| `Optimizer/` | How do teams evolve? |
| `Reporting/` | How do results get written? |
| `Task/` | Which dataset, target and dev/holdout split does the run use? |
| `Models/` | Which models exist, what they cost, and how one is connected (config, rates, credentials, Azure resources, chat model)? |
| `Utils/` | Shared logging helpers. |

## How a run flows through the packages

```
runner.py
  └─ Factories/ (one factory per role, prompts loaded and checked)
  └─ Optimizer/evolver.py  EA
       ├─ Task/task.py            load the CSV, split dev / holdout
       ├─ Knowledge/catalog.py    load the catalog, check it against the cohort
       ├─ the user's SemanticMatcher, Retriever and KnowledgeGraph, built from recipes and probed once
       ├─ Models/model_config.py  models, prices, retry and limit settings
       ├─ Team_Builder/topology.py  which team compositions are allowed
       ├─ ToolResources, TeamGraphBuilder (for check_models), EvaluationRecorder   built once
       ├─ Optimizer/backend.py    RayBackend: starts Ray, sends the run context to the workers once
       │
       └─ each generation:
            Optimizer/reproduction.py   new Teams from parents (Team_Builder/team.py)
            Optimizer/evolver.py        schedules each team's eval_cnt attempts across the workers,
                                        each pinned to a resource by Models/resources.py
            Optimizer/worker.py         one attempt, in a Ray worker process; per worker, the first
                                        attempt of a run builds the matcher, retriever and KG
               Team_Builder/            TeamGraphBuilder + Factories → graph   (agents from Agents/)
               Evaluation/evaluator.py  run graph → features → XGBoost CV → usage → cost; Eval_{i}/ logs
            Evaluation/evaluator.py     EvaluationRecorder, on the driver: team_evaluations.csv row,
                                        provider-abort count
            Optimizer/evolver.py        when a team's last attempt lands: Team.set_evaluation_result(...)
            Optimizer/archive.py        reserve an id per team up front, record it once evaluated
            Optimizer/nsga_tool.py      rank, select survivors
            Reporting/                  CSVs, front plots, frames
       └─ after the last generation:
            Optimizer/evolver.py        final front scored once on the holdout (no LLM)
            Reporting/holdout.py        holdout_results.json + console summary
  └─ Reporting/animations.py  GIFs after the run
```

Dependencies point one way: `Optimizer` → `Evaluation` → `Team_Builder` → `Factories` /
`Models` → `Agents` → `Knowledge` / `Task` / `Utils`. `Reporting` reads results but computes nothing;
it only imports display constants from `Optimizer/objectives.py`. The one two-way package edge is
`Evaluation` ↔ `Reporting`: `evaluator.py` writes each evaluation's logs through
`team_logs.py`, and `EvaluationRecorder` appends `team_evaluations.csv` the same way. `team_logs.py`
reads `Evaluation`'s result types (never `evaluator.py` itself), so no module imports itself back.