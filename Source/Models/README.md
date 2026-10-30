# Models/

Which models exist, what they cost, and how one is connected.

| Module | Contents |
|---|---|
| `model_config.py` | `ModelConfig` / `ModelSpec` from `model_config.json`: models, providers, prices, `param_space`, retry and limit settings. Also `CONNECTION_ARGUMENTS`. |
| `litellm_rates.py` | `litellm_rates(provider, name, deployment)`: per-million rates from litellm, or None. |
| `credentials.py` | `require_credentials`, `load_dotenv_once`, `package_for_provider`. |
| `resources.py` | Numbered Azure resources `R{n}_`: `detect_resource_count`, endpoint/key readers, `foundry_root`. `ResourceAllocator` assigns each in-flight evaluation a resource, so no two share a (resource, model) deployment. |
| `chat_model.py` | `build_chat_model(blueprint, model_config, resource)`, `knob_settings`, `check_param_names`. |

Credentials come only from the environment / `.env`. They never go in kwargs or logs.

## Used by

| Name | File | Use |
|---|---|---|
| `ModelConfig` | `Optimizer/evolver.py` | Loaded once and held; no global copy. |
| | `Optimizer/reproduction.py` | `param_space` for mutation. |
| | `Optimizer/worker.py` | Sent to each Ray worker in the run context. |
| | `Evaluation/evaluator.py` | Retry and recursion limits; `EvaluationRecorder` reads the abort limit. |
| | `Evaluation/usage.py` | Prices each agent's tokens. |
| | `Team_Builder/graph.py`, `Optimizer/preflight.py` | Passed to the model builder. |
| `build_chat_model` | `Team_Builder/graph.py` | Default `build_model` for every agent. |
| `check_param_names` | `Optimizer/evolver.py` | Offline knob check at startup. |
| `knob_settings` | `Optimizer/preflight.py` | Settings `check_models` tries live. |
| `require_credentials`, `detect_resource_count` | `Optimizer/evolver.py` | Startup credential and resource checks. |
| `ResourceAllocator` | `Optimizer/evolver.py` | Pins each in-flight evaluation to a resource. |

## Depends on

`Agents/` (`BluePrint`), `Utils/` (`warn`).
