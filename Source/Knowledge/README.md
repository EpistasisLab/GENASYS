# Knowledge/

The feature catalog and its encoding, plus the contracts for the user's matcher, retriever
and knowledge graph. Agent tools and scoring both read it, so it belongs to neither.

| Module | Contents |
|---|---|
| `catalog.py` | `CatalogEntry`, `load_catalog`, `assert_catalog_matches`, `parse_list_cell`, `normalize_med`, `source_columns`. |
| `features.py` | `build_feature_frame` / `build_feature_frame_with_sources`: catalog features → model columns. |
| `semantic_matcher.py` | `SemanticMatcher` / `Match` contract, `semantic_matcher_tool`, `check_matches`, `MatcherContractError`. |
| `retriever.py` | `Retriever` / `Passage` contract, `retrieve_context_tool`, `check_passages`, `RetrieverContractError`. |
| `knowledge_graph.py` | `KnowledgeGraph` contract and result types (`Node`, `Neighborhood`, `Ranking`, …), `kg_tools`, `check_kg`, `kg_schema_summary`, `KGContractError`. |

A contract violation raises its `*ContractError`, and that stops the run.

## Used by

| Name | File | Use |
|---|---|---|
| `load_catalog`, `assert_catalog_matches` | `Optimizer/evolver.py` | Load the catalog and check it against the cohort at startup. |
| `CatalogEntry` | `Agents/Tools/`, `Evaluation/`, `Optimizer/preflight.py` | The catalog's entry type. |
| `build_feature_frame_with_sources` | `Evaluation/evaluator.py` | Encodes a team's features for scoring. |
| | `Agents/Tools/stats_tools.py` | Encodes features for the statistics, so they see the columns that are scored. |
| `build_feature_frame` | `Optimizer/evolver.py` | Holdout stage. |
| `*_tool`, `kg_tools` | `Factories/base_factory.py` | Builds the agents' tools. |
| `check_matches`, `check_passages`, `check_kg` | `Optimizer/preflight.py` | Startup probes. |
| `*ContractError` | `Evaluation/run_graph.py` | Re-raised instead of scoring the team 0. |
| `SemanticMatcher`, `Retriever`, `KnowledgeGraph` | `Agents/Tools/tool_resources.py`, `Optimizer/` | Built from recipes (`make_matcher`, `make_retriever`, `make_kg`): once in the driver, where they are probed, and once in each Ray worker process (`Optimizer/worker.py`). |
| `kg_schema_summary` | `Team_Builder/graph.py` | Fills `{kg_schema_summary}` in prompts. |


## Depends on

Nothing in `Source/`.
