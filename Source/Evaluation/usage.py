"""Token usage of a finished run, per agent, and what it cost.

``aggregate_usage`` reads ``usage_metadata`` off every message of a run
(``GraphRun.all_messages``: the team's messages plus the worker turns that never reach
the team's state) and sums it by the agent name LangGraph stamps on each message; ``price_usage`` prices each agent's
tokens at its own model's rates (``ModelConfig.price_agent``). The total tokens or the
total dollars become the team's objective_2.

Every token-bearing message must name a roster agent. ``graph.py`` gives each agent its
role name, so a message with no name, or one naming an agent the team does not have, is
our bug; pricing it at some fallback model would put a wrong number on the Pareto front.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from typeguard import typechecked

from ..Models.model_config import ModelConfig

# Our key -> the names providers report it under (LangChain's first, then OpenAI's raw).
_COUNT_KEYS = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
}


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one message, one agent or a whole run.

    ``cached_input_tokens`` is the cache-read part of ``input_tokens``, not an extra
    amount; it is billed at the model's cached rate.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(self.input_tokens + other.input_tokens,
                          self.output_tokens + other.output_tokens,
                          self.total_tokens + other.total_tokens,
                          self.cached_input_tokens + other.cached_input_tokens)


NO_USAGE = TokenUsage(0, 0, 0, 0)


@dataclass(frozen=True)
class RunUsage:
    """``per_agent`` is keyed by agent name; agents that never called a model are absent."""

    per_agent: Dict[str, TokenUsage]
    total: TokenUsage


@dataclass(frozen=True)
class AgentCost:
    model: str
    usage: TokenUsage
    cost_usd: float


@dataclass(frozen=True)
class RunCost:
    """A run's cost and tokens, per agent and in total.

    ``usage.total_tokens`` and ``total_usd`` are the two candidates for objective_2
    (``token_count`` / ``cost_usd``), so either can be optimized from this one result.
    """

    per_agent: Dict[str, AgentCost]
    usage: TokenUsage
    total_usd: float


@typechecked
def normalize_usage(usage: Mapping[str, Any]) -> TokenUsage:
    """One message's ``usage_metadata`` as a ``TokenUsage``.

    Input and output counts are required under either the LangChain or the OpenAI
    name. ``total_tokens`` defaults to their sum. The cache-read count is optional
    (Ollama and providers without prompt caching never report one, and there it truly
    is 0) and is capped at ``input_tokens``.
    """
    counts: Dict[str, int] = {}
    for ours, names in _COUNT_KEYS.items():
        present = [n for n in names if n in usage]
        assert present, f"usage_metadata has no {' / '.join(names)}: {dict(usage)}"
        counts[ours] = int(usage[present[0]])
        assert counts[ours] >= 0, f"negative {present[0]} in usage_metadata: {dict(usage)}"
    total = int(usage["total_tokens"]) if "total_tokens" in usage \
        else counts["input_tokens"] + counts["output_tokens"]
    details = usage.get("input_token_details") or {}
    cached = int(details.get("cache_read") or 0)
    return TokenUsage(input_tokens=counts["input_tokens"],
                      output_tokens=counts["output_tokens"],
                      total_tokens=total,
                      cached_input_tokens=min(cached, counts["input_tokens"]))


@typechecked
def aggregate_usage(messages: Sequence[Any]) -> RunUsage:
    """Sum the usage of a run's messages per agent name.

    ``messages`` must hold each model call once (``GraphRun.all_messages``). Messages
    without ``usage_metadata`` (tool results, handoffs, the kickoff) are skipped. Raises
    if a token-bearing message carries no agent name.
    """
    per_agent: Dict[str, TokenUsage] = {}
    for message in messages:
        metadata = getattr(message, "usage_metadata", None)
        if not metadata:
            continue
        assert message.name, \
            (f"a {type(message).__name__} carries token usage but no agent name, so its "
             f"cost cannot be attributed; every agent in graph.py must be named. "
             f"Content starts: {str(message.content)[:80]!r}")
        per_agent[message.name] = per_agent.get(message.name, NO_USAGE) + normalize_usage(metadata)
    total = sum(per_agent.values(), NO_USAGE)
    return RunUsage(per_agent=per_agent, total=total)


@typechecked
def price_usage(usage: RunUsage, role_to_model: Dict[str, str],
                config: ModelConfig) -> RunCost:
    """Price each agent's tokens at its own model's rates.

    Args:
        usage: from ``aggregate_usage``.
        role_to_model: the team's roster, agent name -> model name.
        config: the run's ``ModelConfig``, which holds the rates.

    Raises:
        AssertionError: usage under a name that is not on the roster.
    """
    strangers = sorted(set(usage.per_agent) - set(role_to_model))
    assert not strangers, \
        (f"token usage under {strangers}, which are not on the team's roster "
         f"{sorted(role_to_model)}; message names must be the roster's role names")
    per_agent = {
        role: AgentCost(model=role_to_model[role], usage=tokens,
                        cost_usd=config.price_agent(role_to_model[role], tokens.input_tokens,
                                                    tokens.output_tokens,
                                                    tokens.cached_input_tokens))
        for role, tokens in usage.per_agent.items()
    }
    return RunCost(per_agent=per_agent, usage=usage.total,
                   total_usd=sum(cost.cost_usd for cost in per_agent.values()))
