"""Sort a run's factory list into the Supervisor slot and the workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .base_factory import AgentFactory
from .single_agent_factory import SingleAgentFactory
from .supervisor_factory import SupervisorFactory


@dataclass(frozen=True)
class FactorySet:
    # Leads teams that have workers.
    supervisor: Optional[SupervisorFactory]
    # Fills the Supervisor slot on a team with no workers.
    single_agent: Optional[SingleAgentFactory]
    # role_name -> factory, for every worker role the run can field.
    workers: Dict[str, AgentFactory]


def split_factories(factories: Sequence[AgentFactory]) -> FactorySet:
    """Sort ``factories`` by the slot each fills; fails on a duplicate.

    At most one ``SupervisorFactory`` and one ``SingleAgentFactory`` (both fill
    the Supervisor slot, on teams with and without workers), and one factory per
    worker role. Whether the ones that are optional here are actually needed
    depends on the topology rules; the EA checks that.
    """
    supervisor: Optional[SupervisorFactory] = None
    single_agent: Optional[SingleAgentFactory] = None
    workers: Dict[str, AgentFactory] = {}
    for factory in factories:
        name = type(factory).__name__
        if isinstance(factory, SupervisorFactory):
            assert supervisor is None, "More than one SupervisorFactory was supplied."
            supervisor = factory
        elif isinstance(factory, SingleAgentFactory):
            assert single_agent is None, "More than one SingleAgentFactory was supplied."
            single_agent = factory
        else:
            assert factory.call_tool_name, (
                f"{name} has no call_tool_name, so the Supervisor could never hand "
                f"off to it; only worker factories belong alongside the Supervisor's.")
            assert factory.role_name not in workers, (
                f"Two factories for the worker role '{factory.role_name}': "
                f"{type(workers[factory.role_name]).__name__} and {name}.")
            workers[factory.role_name] = factory
    return FactorySet(supervisor=supervisor, single_agent=single_agent, workers=workers)
