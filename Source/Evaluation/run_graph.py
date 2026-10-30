"""Run a team's compiled graph and record how it stopped.

``invoke_graph`` never raises for an ``Exception``: a failed run still returns the last
state it reached, with the error, its type and the full traceback alongside in a
``GraphRun``. The evaluator writes the traceback to that team's ``run_history.log`` and
scores the team 0; only a one-line summary goes to the console. ``KeyboardInterrupt``
and other ``BaseException``s are not caught, so Ctrl-C still stops the run. Neither is
``MatcherContractError``, ``RetrieverContractError`` or ``KGContractError``: a semantic
matcher returning features the catalog lacks, or a retriever or knowledge graph breaking
its contract, is a bug in the run's setup, not something the team did, so it stops the
run.

Workers' own turns: langgraph-supervisor hands a worker's final message back to the
team, and only that message ever reaches the team's state; the worker's tool calls and
tool results are dropped. So the graph is streamed with ``subgraphs=True`` and every
agent's own state is captured as it runs. ``GraphRun.handoffs`` keeps each worker
handoff's full transcript, for the logs, and ``all_messages`` adds every model call to
the team's messages, so a worker's tool-calling turns are priced too. A worker's model is shown only its handoff
(``Team_Builder/handoff.py``), but its state still holds the team history it was handed,
so the capture below is unaffected. Both survive a
crash mid-handoff, which a worker's state never would: it reaches the team only when
the worker finishes.

``is_infrastructure_error`` separates the provider failing (rate limit, bad key, outage)
from the team failing. Both stop the graph, but only the first says nothing about the
team: the evaluator retries those, and the EA aborts after too many in a row, so an
expired key cannot read as "every team this generation was terrible".
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, ToolMessage

from ..Agents.state import initial_state
from ..Knowledge.knowledge_graph import KGContractError
from ..Knowledge.retriever import RetrieverContractError
from ..Knowledge.semantic_matcher import MatcherContractError
from ..Team_Builder.handoff import handoff_message, handoff_task
from ..Team_Builder.topology import SUPERVISOR

# HTTP statuses that are the provider's problem or a run-wide configuration problem:
# 401/403 wrong or revoked key, 404 model or deployment not found, 408 timeout,
# 409 conflict, 429 rate limit or quota, 5xx the provider's own failure (529 overloaded).
# Every other status is the team's request being refused, e.g. 400 for a context that
# grew past the model's limit, a content-filter rejection or a malformed tool call.
_INFRA_STATUS_CODES = frozenset({401, 403, 404, 408, 409, 429, 500, 502, 503, 504, 529})

# Exceptions with no HTTP status that mean the request never got an answer. Matched by
# class name anywhere in the exception's class hierarchy, so this module imports no
# provider package (an Ollama-only install never needs openai). Base classes such as
# openai's ``APIError`` / ``APIStatusError`` must NOT be listed: every openai HTTP error
# inherits from them, including a 400 the team caused.
_INFRA_ERROR_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError",                       # openai
    "ConnectError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError",  # httpx
    "Timeout", "TimeoutError",
})

# Ollama errors that arrive with a 5xx status but are the model's output going wrong, not
# the server: a malformed tool call, or a reply stuck repeating one token. Each is the
# team's result, so it is scored 0 rather than retried; counted as provider failures they
# aborted gpt-oss:20b's run after 10 such teams in a row.
_TEAM_ERROR_MESSAGES = ("error parsing tool call", "token repeat limit reached")


class TeamTimeLimit(Exception):
    """The run was still going after the config's ``team_time_limit_s``. Not a
    ``TimeoutError`` on purpose: ``is_infrastructure_error`` matches that name, and a team
    that never finishes is the team's result, not the provider failing, so it must not be
    retried."""


@dataclass(frozen=True)
class Handoff:
    """One worker handoff: what the worker was asked and everything it did.

    ``messages`` are only the ones the worker added (its model turns, tool calls with
    their arguments, and tool results), not the team history it was handed.
    ``worker_input`` is the one message its model was shown in place of that history
    (``handoff_message``). ``completed`` is False when the run stopped while the worker
    was still working.
    """

    agent: str
    # Position among all of the run's handoffs, and among this agent's.
    index: int
    call_number: int
    # The Supervisor's handoff call, its task argument, and the content of the message
    # that made it.
    tool_call_id: str
    task: str
    supervisor_message: Any
    worker_input: str
    messages: List[Any]
    completed: bool


@dataclass(frozen=True)
class GraphRun:
    """How one run of a team's graph ended.

    ``state`` is the last state the graph reached: the final one for a clean run, the
    last one before the exception otherwise. ``handoffs`` are the worker handoffs, in
    the order they started, and ``agent_messages`` every message each agent's own
    state held (see ``all_messages``). The error fields are all None (and
    ``infrastructure_error`` False) for a clean run. ``error`` is one line
    (``"RateLimitError: ..."``), ``error_type`` the exception's class name (e.g.
    ``GraphRecursionError`` when the team ran out of ``recursion_limit``), and
    ``traceback`` the full text for the team's ``run_history.log``.
    """

    state: Dict[str, Any]
    error: Optional[str]
    error_type: Optional[str]
    traceback: Optional[str]
    infrastructure_error: bool
    handoffs: List[Handoff]
    agent_messages: List[Any]

    def all_messages(self) -> List[Any]:
        """The team's messages, then any message only an agent's own state held, each
        once (by id): every model call of the run, wherever its message ended up."""
        seen = set()
        out = []
        for message in [*self.state["messages"], *self.agent_messages]:
            assert message.id, f"a {type(message).__name__} reached the run with no id"
            if message.id not in seen:
                seen.add(message.id)
                out.append(message)
        return out


def _status_code(exc: BaseException) -> Optional[int]:
    """The HTTP status an exception carries (openai, ollama, LiteLLM), if any."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_infrastructure_error(exc: BaseException) -> bool:
    """Is this the provider failing, rather than the team?

    An exception with an HTTP status is decided by the status alone
    (``_INFRA_STATUS_CODES``); one without is decided by its class name
    (``_INFRA_ERROR_NAMES``). The exception's direct ``__cause__`` and ``__context__``
    are checked too, since client libraries often wrap the transport error. A message in
    ``_TEAM_ERROR_MESSAGES`` anywhere in that chain overrides every status.
    """
    chain = [e for e in (exc, exc.__cause__, exc.__context__) if e is not None]
    if any(text in str(e) for e in chain for text in _TEAM_ERROR_MESSAGES):
        return False
    for e in chain:
        status = _status_code(e)
        if status is not None:
            if status in _INFRA_STATUS_CODES:
                return True
            continue
        if any(klass.__name__ in _INFRA_ERROR_NAMES for klass in type(e).__mro__):
            return True
    return False


def invoke_graph(app: Any, user_message: str, recursion_limit: int,
                 time_limit_s: Optional[float] = None) -> GraphRun:
    """Stream a compiled team graph from its kickoff message; return how it ended.

    Args:
        app: the compiled LangGraph app.
        user_message: the lead agent's rendered kickoff message.
        recursion_limit: cap on graph super-steps for the run; required (the caller
            passes ``ModelConfig.recursion_limit``) so a forgotten argument cannot fall
            back to LangGraph's far smaller default.
        time_limit_s: wall-clock cap on the run (``ModelConfig.team_time_limit_s``);
            None for no cap. Checked after every graph step, so one model call already
            in flight still finishes (``request_timeout_s`` bounds it). Running out ends
            the run as ``TeamTimeLimit``.
    """
    last_state: Dict[str, Any] = initial_state(user_message)
    # Each agent invocation's namespace -> (messages it was given, its latest messages).
    agents: Dict[Tuple[str, ...], Tuple[int, List[Any]]] = {}
    error: Optional[Exception] = None
    trace: Optional[str] = None
    deadline = None if time_limit_s is None else time.monotonic() + time_limit_s
    try:
        for namespace, state in app.stream(last_state,
                                           config={"recursion_limit": recursion_limit},
                                           stream_mode="values", subgraphs=True):
            if not namespace:
                last_state = state
            else:
                given = agents[namespace][0] if namespace in agents else len(state["messages"])
                agents[namespace] = (given, list(state["messages"]))
            if deadline is not None and time.monotonic() > deadline:
                raise TeamTimeLimit(f"still running after team_time_limit_s "
                                    f"({time_limit_s:g} s)")
    except (MatcherContractError, RetrieverContractError, KGContractError):
        raise
    except Exception as e:
        error, trace = e, traceback.format_exc()

    infrastructure_error = error is not None and is_infrastructure_error(error)
    if error is not None:
        kind = "PROVIDER FAILURE" if infrastructure_error else "graph error"
        print(f"    {kind} stopped the run early ({type(error).__name__}): {error}",
              flush=True)
    state = dict(last_state)
    return GraphRun(state=state,
                    error=None if error is None else f"{type(error).__name__}: {error}",
                    error_type=None if error is None else type(error).__name__,
                    traceback=trace, infrastructure_error=infrastructure_error,
                    handoffs=_handoffs(agents, state),
                    agent_messages=[m for _, messages in agents.values() for m in messages])


def _handoffs(agents: Dict[Tuple[str, ...], Tuple[int, List[Any]]],
              team_state: Dict[str, Any]) -> List[Handoff]:
    """The worker handoffs among the captured agent invocations, in the order they started.

    A namespace is ``("{node}:{task id}",)``, new for every invocation. The Supervisor's
    own invocations are not handoffs: all of its messages reach the team's state. A
    worker's input ends with the Supervisor's handoff call and its tool result; it
    completed if its last message reached the team's state.
    """
    team_ids = {m.id for m in team_state["messages"]}
    handoffs: List[Handoff] = []
    calls: Dict[str, int] = {}
    for namespace, (given, messages) in agents.items():
        agent = namespace[0].split(":", 1)[0]
        if len(namespace) != 1 or agent == SUPERVISOR:
            continue
        call_message, result = messages[given - 2], messages[given - 1]
        assert isinstance(call_message, AIMessage) and isinstance(result, ToolMessage), \
            (f"{agent} was started without a handoff call: its input ends with "
             f"{type(call_message).__name__}, {type(result).__name__}")
        own = messages[given:]
        handoffs.append(Handoff(agent=agent, index=len(handoffs),
                                call_number=calls.get(agent, 0),
                                tool_call_id=result.tool_call_id,
                                task=handoff_task(call_message, result.tool_call_id),
                                supervisor_message=call_message.content,
                                worker_input=handoff_message(call_message,
                                                             result.tool_call_id).content,
                                messages=own,
                                completed=bool(own) and own[-1].id in team_ids))
        calls[agent] = calls.get(agent, 0) + 1
    return handoffs
