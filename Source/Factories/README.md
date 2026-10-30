# Factories/

One factory per agent role. A factory owns its role's prompt templates and picks its tools.
It builds no LLM and no graph.

| Module | Contents |
|---|---|
| `base_factory.py` | `AgentFactory`: loads, checks and renders the templates; `tools(team, resources)` from class flags; `review_fields`; `preflight()` hook. `team_placeholders`, `team_review_fields`, `WORKER_SYSTEM_PLACEHOLDERS`. |
| `supervisor_factory.py` | `SupervisorFactory`, the lead of a team with workers. |
| `single_agent_factory.py` | `SingleAgentFactory`, the Supervisor slot with no workers. |
| `rag_factory.py`, `kg_factory.py`, `statistician_factory.py`, `domain_expert_factory.py` | The worker roles. |
| `factory_set.py` | `FactorySet(supervisor, single_agent, workers)` and `split_factories`. |

Templates are `{role}_system.md` / `{role}_user.md` in `--prompt_dir`.

## Used by

| File | Uses |
|---|---|
| `runner.py` | Builds each role factory from its two templates. |
| `Optimizer/evolver.py` | `split_factories` turns the runner's list into the run's `FactorySet`. |
| `Optimizer/preflight.py` | `FactorySet`, `AgentFactory`: finds factories in use and checks the topology is buildable. Reads `retrieves` / `searches_kg` to decide whether a retriever or KG is required. |
| `Team_Builder/graph.py` | Picks the roster's factories from the `FactorySet`. Calls `tools()`, `render()`, `team_placeholders` and `team_review_fields` to build each agent. |

## Depends on

`Agents/Tools/` (tool builders), `Knowledge/` (matcher, retriever and KG tools).
