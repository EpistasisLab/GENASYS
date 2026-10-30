"""Compile a team into the LangGraph app the Evaluator runs.

``TeamGraphBuilder`` is set up once per run with everything that does not change from
team to team (the factories, the tool resources, the task, the model config) and is
called once per evaluation attempt with a ``Team``; every call builds new models, tools
and a new graph, so no attempt inherits another's state.

The team's roster decides the shape:

- A Supervisor with workers: each worker is a ``create_agent`` ReAct subgraph, and the
  Supervisor reaches each one only through that worker factory's ``call_*`` handoff tool,
  which carries a ``task``; a worker's model sees only its handoff and its own turns, not
  the team's history (see ``handoff.py``).
  langgraph-supervisor adds its own ``transfer_to_*`` tools only when it is given no
  handoff tools, so supplying ours means the Supervisor's tools are exactly its
  feature-state tools plus one ``call_*`` per worker on this team; this is checked.
- The Supervisor slot alone: the single agent, built by the same ``create_supervisor``
  call with no workers and so no handoff tools. It is not built as a bare agent because
  ``return_final_feature_set`` ends the run with ``Command.PARENT``, which needs a
  parent graph.

Every agent is named by its role, which stamps its messages for pricing.

A built team is a ``TeamGraph``; the Evaluator takes any ``GraphBuilder`` (this one, or a
scripted one in tests).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph_supervisor import create_supervisor
from typeguard import typechecked

from ..Agents.blueprint import BluePrint
from ..Models.chat_model import build_chat_model
from ..Agents.state import TeamState, WorkerState
from ..Agents.Tools.tool_resources import ToolResources
from ..Task.task import Task
from ..Factories.base_factory import AgentFactory, team_placeholders, team_review_fields
from ..Factories.factory_set import FactorySet
from ..Knowledge.knowledge_graph import kg_schema_summary
from ..Models.model_config import ModelConfig
from .handoff import HandoffInput, create_task_handoff_tool
from .team import Team
from .topology import SUPERVISOR

# Builds an agent's chat model from its blueprint.
ModelBuilder = Callable[[BluePrint], BaseChatModel]


@dataclass(frozen=True)
class TeamGraph:
    """A team, compiled and ready to run.

    ``review_fields`` are the team's ``*_review`` fields (``team_review_fields``), which
    every feature entry must carry. ``role_to_model`` maps every agent name that appears
    on the run's messages to its model, for pricing.
    """

    app: Any
    kickoff_message: str
    review_fields: Dict[str, str]
    role_to_model: Dict[str, str]


# Builds a fresh graph for a team; called once per attempt, so no attempt inherits
# another's state.
GraphBuilder = Callable[[Team], TeamGraph]


class _SupervisorRemainingStepsFilter(logging.Filter):
    """Drops LangGraph's warning that the Supervisor wrote ``remaining_steps``.

    langgraph-supervisor uses one state schema for the Supervisor's inner ReAct agent,
    which requires ``remaining_steps``, and for the team graph around it, where that
    managed value has no channel. Every Supervisor turn returns its whole state, so
    LangGraph warns that it ignored the write; nothing is lost. Any other "unknown
    channel" warning still shows: it would be a real bug, such as a tool writing a
    field the state does not have.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (message.startswith(f"Task {SUPERVISOR} with path ")
                    and message.endswith("wrote to unknown channel remaining_steps, "
                                         "ignoring it."))


def silence_supervisor_remaining_steps_warning() -> None:
    """Install the filter on LangGraph's logger, once per process. LangGraph logs the
    warning on its package logger, ``"langgraph"``, not on the module's own."""
    logger = logging.getLogger("langgraph")
    if not any(isinstance(f, _SupervisorRemainingStepsFilter) for f in logger.filters):
        logger.addFilter(_SupervisorRemainingStepsFilter())


class TeamGraphBuilder:
    """Builds a fresh ``TeamGraph`` for a team; the Evaluator's ``build_graph``."""

    @typechecked
    def __init__(self, factories: FactorySet, resources: ToolResources, task: Task,
                 model_config: ModelConfig, resource: int,
                 build_model: Optional[ModelBuilder] = None) -> None:
        """
        Args:
            factories: the run's factories, sorted into slots (``split_factories``).
            resources: the run's tool data, shared by every team.
            task: names the target and the data file in the prompts.
            resource: the Azure ``R{n}_`` resource every model connects through.
            build_model: builds a model from a blueprint; ``build_chat_model`` by
                default, replaced in tests.
        """
        self.factories = factories
        self.resources = resources
        self.task_subs = {"target_variable": task.target_column,
                          "file_name": os.path.basename(task.csv_path)}
        # What the graph holds, for the prompts of the roles that search it. Without a
        # graph no such role runs, so rendering one fails on the missing placeholder.
        if resources.kg is not None:
            self.task_subs["kg_schema_summary"] = kg_schema_summary(resources.kg.schema())
        self.build_model: ModelBuilder = build_model or (
            lambda blueprint: build_chat_model(blueprint, model_config, resource))
        silence_supervisor_remaining_steps_warning()

    @typechecked
    def __call__(self, team: Team) -> TeamGraph:
        roster = team.get_roster()
        hub, workers = self.team_factories(roster)
        everyone: List[AgentFactory] = [hub, *workers]

        compiled_workers = [
            create_agent(model=self.build_model(roster[w.role_name]),
                         tools=w.tools(everyone, self.resources), name=w.role_name,
                         system_prompt=self._render(w, everyone, "system") + "\n\n"
                         + self._render(w, everyone, "user"),
                         state_schema=WorkerState, middleware=[HandoffInput(w.role_name)])
            for w in workers]

        handoffs = [create_task_handoff_tool(agent_name=w.role_name, name=w.call_tool_name,
                                             description=w.description) for w in workers]
        hub_tools = hub.tools(everyone, self.resources)
        clash = sorted({t.name for t in hub_tools} & {h.name for h in handoffs})
        assert not clash, f"handoff tools {clash} clash with the Supervisor's own tools"

        app = create_supervisor(agents=compiled_workers,
                                model=self.build_model(roster[SUPERVISOR]),
                                tools=[*hub_tools, *handoffs],
                                prompt=self._render(hub, everyone, "system"),
                                state_schema=TeamState,
                                supervisor_name=SUPERVISOR).compile()
        return TeamGraph(app=app, kickoff_message=self._render(hub, everyone, "user"),
                         review_fields=team_review_fields(everyone),
                         role_to_model={role: bp.get_model() for role, bp in roster.items()})

    def team_factories(self, roster: Dict[str, BluePrint]
                       ) -> Tuple[AgentFactory, List[AgentFactory]]:
        """The factory filling the Supervisor slot, and the workers' factories in the
        order the run supplied them. Fails if a role on the roster has no factory."""
        assert SUPERVISOR in roster, f"a team needs the '{SUPERVISOR}' slot: {sorted(roster)}"
        roles = set(roster) - {SUPERVISOR}
        missing = sorted(roles - set(self.factories.workers))
        assert not missing, (f"no factory for the worker role(s) {missing} on this team; "
                             f"the run has {sorted(self.factories.workers)}")
        hub = self.factories.supervisor if roles else self.factories.single_agent
        assert hub is not None, (
            f"a team {'with' if roles else 'without'} workers needs a "
            f"{'SupervisorFactory' if roles else 'SingleAgentFactory'}, and the run has none")
        workers = [f for role, f in self.factories.workers.items() if role in roles]
        return hub, workers

    def _render(self, me: AgentFactory, team: Sequence[AgentFactory], key: str) -> str:
        subs = {**self.task_subs, **team_placeholders(me, team)}
        return me.render(key, subs)
