"""The Supervisor's ``call_*`` handoff tools, and what a worker's model is shown.

langgraph-supervisor's own handoff tool forwards the team's whole message history to
the worker, ending with the Supervisor's turn that made the call and its "Successfully
transferred" result. A worker's model reads that turn as one it wrote itself: it
believes it has already asked for the work and waits for the answer, so it hands back
having done nothing. Re-reading the whole team history on every handoff is also where
most of a run's tokens went.

So the graph is unchanged, but a worker's model sees only its handoff:

- ``create_task_handoff_tool`` is langgraph-supervisor's tool with one more argument,
  ``task``: what the Supervisor wants from this call. Its messages and its routing are
  the same, so the team's state, the transcripts and the logs record exactly what they
  did before.
- ``HandoffInput`` is middleware on every worker. Before each model call it replaces
  everything up to and including the handoff's tool result with one message from the
  Supervisor (``handoff_message``), keeping the worker's own turns after it. Only the
  request sent to the model changes; the worker's state keeps every message.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence, Tuple

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from langgraph_supervisor.handoff import (METADATA_KEY_HANDOFF_DESTINATION,
                                          _remove_non_handoff_tool_calls)
from typing_extensions import Annotated

from .topology import SUPERVISOR

TASK_DESCRIPTION = ("What you want from this teammate on this call: whether to generate "
                    "candidates, review named pending features, or both, and the concepts "
                    "or feature_name strings involved. The teammate sees this and your "
                    "message on this turn, not the rest of the run's conversation.")


def create_task_handoff_tool(agent_name: str, name: str, description: str) -> BaseTool:
    """A ``call_*`` tool that hands control to ``agent_name`` with a ``task``.

    Mirrors langgraph-supervisor's ``create_handoff_tool`` with handoff messages on:
    a single call jumps to the worker with the history plus the tool result; parallel
    calls each ``Send`` the history with only their own call kept on the last message.
    """

    @tool(name, description=description)
    def handoff_to_agent(task: Annotated[str, TASK_DESCRIPTION],
                         state: Annotated[dict, InjectedState],
                         tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}", name=name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name})
        last_ai_message = state["messages"][-1]
        if len(last_ai_message.tool_calls) > 1:
            messages = [*state["messages"][:-1],
                        _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                        tool_message]
            return Command(graph=Command.PARENT,
                           goto=[Send(agent_name, {**state, "messages": messages})])
        return Command(goto=agent_name, graph=Command.PARENT,
                       update={**state, "messages": [*state["messages"], tool_message]})

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent


def find_handoff(messages: Sequence[AnyMessage], agent_name: str
                 ) -> Tuple[int, AIMessage, str]:
    """The latest handoff to ``agent_name`` in ``messages``: the index of its tool
    result, the Supervisor's message that made the call, and the call's id. Fails if
    there is none: a worker only ever runs from a handoff."""
    for i in range(len(messages) - 1, 0, -1):
        result = messages[i]
        if (isinstance(result, ToolMessage)
                and result.response_metadata.get(METADATA_KEY_HANDOFF_DESTINATION) == agent_name):
            call = messages[i - 1]
            assert isinstance(call, AIMessage) and any(
                c["id"] == result.tool_call_id for c in call.tool_calls), \
                f"the handoff result to {agent_name} does not follow its call"
            return i, call, result.tool_call_id
    raise AssertionError(f"{agent_name} is running without a handoff in its messages")


def message_text(content: Any) -> str:
    """The text of a message's content: a string, or the text blocks of a block list
    (an Anthropic turn that calls tools carries its prose in a ``text`` block)."""
    if isinstance(content, str):
        return content.strip()
    return "\n".join(b["text"] for b in content
                     if isinstance(b, dict) and b.get("type") == "text").strip()


def handoff_task(call: AIMessage, tool_call_id: str) -> str:
    """The ``task`` argument of the handoff call ``tool_call_id`` on ``call``."""
    [args] = [c["args"] for c in call.tool_calls if c["id"] == tool_call_id]
    return str(args.get("task", "")).strip()


def handoff_message(call: AIMessage, tool_call_id: str) -> HumanMessage:
    """What a worker's model is given in place of the team history: the task, then the
    Supervisor's own words on that turn (its JSON object), if it wrote any."""
    task = handoff_task(call, tool_call_id) or "(the Supervisor gave no task)"
    parts = [f"Task from the Supervisor:\n{task}"]
    said = message_text(call.content)
    if said:
        parts.append(f"The Supervisor's message:\n{said}")
    return HumanMessage(content="\n\n".join(parts), name=SUPERVISOR)


def worker_view(messages: Sequence[AnyMessage], agent_name: str) -> List[AnyMessage]:
    """``messages`` as ``agent_name``'s model should see them: its handoff message, then
    every turn it has taken since."""
    i, call, tool_call_id = find_handoff(messages, agent_name)
    return [handoff_message(call, tool_call_id), *messages[i + 1:]]


class HandoffInput(AgentMiddleware):
    """Shows a worker's model only its handoff and its own turns (see the module)."""

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request.override(messages=worker_view(request.messages,
                                                              self.agent_name)))
