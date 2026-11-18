"""The run's model-call budget: one count shared by every agent on a team.

A run may invoke models at most ``model_call_limit`` times in total, whoever makes the
call: the Supervisor, a worker or the single agent. The budget is the same unit for a
team and for the single agent, so the two compete on equal terms. A provider client's
own retries happen inside one invocation and count once.

Every agent is shown the count before each of its model calls, as a message saved to
its messages, so the count is in the run log and the worker transcripts. The call that
would go past the limit raises ``ModelCallLimit`` instead, which stops the whole run;
the evaluator records it as truncated, an unscored run that is not retried.

- ``ModelCallBudget`` is the counter. ``TeamGraphBuilder`` makes a new one for every
  graph it builds, so no attempt inherits another's count.
- ``budget_hook`` is the Supervisor's (and the single agent's) ``pre_model_hook``.
- ``WorkerBudget`` is the same for a worker, as middleware. It also turns off parallel
  tool calls, so a worker, like the Supervisor and the single agent, makes at most one
  tool call per model call: a model call then means the same work for every agent.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

# Name stamped on every budget message, so the logs can tell it from a real turn.
BUDGET_MESSAGE_NAME = "model_call_budget"


class ModelCallLimit(Exception):
    """The run asked for a model call after using all ``model_call_limit`` calls. Like
    ``TeamTimeLimit``, the team's result, not the provider failing, so it is never
    retried."""


class ModelCallBudget:
    """How many model calls the run has made, out of ``limit``."""

    def __init__(self, limit: int) -> None:
        assert limit > 0, f"model_call_limit must be positive: {limit}"
        self.limit = limit
        self.used = 0

    def spend(self, agent: str) -> HumanMessage:
        """Count one model call by ``agent``; the message telling it the calls left.

        Raises ``ModelCallLimit`` when every call has been used: the run stops before
        the call is made.
        """
        if self.used >= self.limit:
            raise ModelCallLimit(f"{agent} asked for a model call after the run had used "
                                 f"all {self.limit} (model_call_limit)")
        self.used += 1
        left = self.limit - self.used
        return HumanMessage(content=f"Model calls left in this run after this one: "
                                    f"{left} of {self.limit}.",
                            name=BUDGET_MESSAGE_NAME)


def budget_hook(budget: ModelCallBudget, agent: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """A ``pre_model_hook`` that spends one call of ``budget`` before each of ``agent``'s
    model calls and saves the count to its messages."""

    def hook(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [budget.spend(agent)]}

    return hook


class WorkerBudget(AgentMiddleware):
    """A worker's share of the run's budget: spends one call before each of its model
    calls, saves the count to its messages, and binds its tools without parallel tool
    calls."""

    def __init__(self, budget: ModelCallBudget, agent_name: str) -> None:
        super().__init__()
        self.budget = budget
        self.agent_name = agent_name

    def before_model(self, state: Any, runtime: Any) -> Dict[str, Any]:
        return {"messages": [self.budget.spend(self.agent_name)]}

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        settings = {**request.model_settings, "parallel_tool_calls": False}
        return handler(request.override(model_settings=settings))
