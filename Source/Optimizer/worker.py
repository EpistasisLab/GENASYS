"""What a Ray worker runs: one evaluation attempt of one team, with the run's tools.

Every attempt builds its team's graph from the run's ``ToolResources``: the matcher, the
retriever, the knowledge graph, the dev rows and the catalog. Ray sends the run's
``RunContext`` to each worker once, but not those tools: the retriever cannot be pickled,
and the matcher carries a whole language model. So each worker process builds its own
tools from the recipes in the context, the first time it runs an attempt of that run, and
keeps them for the rest of the run.

The retriever's recipe is handed a folder of the process's own, and copies the Chroma store
into it. Opening a Chroma store rewrites its ``chroma.sqlite3``, so no two processes share
a file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import ray
from typeguard import typechecked

from ..Agents.Tools.tool_resources import ToolResources
from ..Evaluation.evaluator import Evaluator
from ..Evaluation.result import EvalResult
from ..Factories.factory_set import FactorySet
from ..Knowledge.catalog import CatalogEntry
from ..Knowledge.knowledge_graph import KnowledgeGraph
from ..Knowledge.retriever import Retriever
from ..Knowledge.semantic_matcher import SemanticMatcher
from ..Models.model_config import ModelConfig
from ..Task.task import Task
from ..Team_Builder.graph import ModelBuilder, TeamGraphBuilder
from ..Team_Builder.team import Team


@dataclass(frozen=True)
class RunContext:
    """Everything an attempt needs from its run, sent to each worker once.

    The three ``make_`` fields are the tool recipes the EA was given. ``make_retriever``
    takes a folder the process may copy the store into; the other two take no arguments.
    """

    run_key: str
    run_dir: str
    task: Task
    catalog: Dict[str, CatalogEntry]
    factories: FactorySet
    model_config: ModelConfig
    cv_folds: int
    seed: int
    make_matcher: Callable[[], SemanticMatcher]
    make_retriever: Optional[Callable[[str], Retriever]]
    make_kg: Optional[Callable[[], KnowledgeGraph]]
    build_model: Optional[ModelBuilder]


def store_dir(run_dir: str, owner: str) -> str:
    """The folder ``owner`` (the driver, or one worker process) may copy the store into."""
    return os.path.join(run_dir, "worker_stores", owner)


def build_tools(make_matcher: Callable[[], SemanticMatcher],
                make_retriever: Optional[Callable[[str], Retriever]],
                make_kg: Optional[Callable[[], KnowledgeGraph]],
                work_dir: str) -> Tuple[SemanticMatcher, Optional[Retriever],
                                        Optional[KnowledgeGraph]]:
    """The run's three tool objects, built from their recipes in this process."""
    matcher = make_matcher()
    retriever = make_retriever(work_dir) if make_retriever is not None else None
    kg = make_kg() if make_kg is not None else None
    return matcher, retriever, kg


# The tools this process built, for the run it is serving. A process serves one run at a
# time, so a new run replaces the old tools rather than adding to them.
_TOOLS: Dict[str, ToolResources] = {}


def _resources(run: RunContext) -> ToolResources:
    if run.run_key not in _TOOLS:
        _TOOLS.clear()
        matcher, retriever, kg = build_tools(
            run.make_matcher, run.make_retriever, run.make_kg,
            store_dir(run.run_dir, f"pid_{os.getpid()}"))
        _TOOLS[run.run_key] = ToolResources(
            catalog=run.catalog, matcher=matcher, dataset=run.task.get_dataframe(),
            target_column=run.task.target_column, retriever=retriever, kg=kg)
    return _TOOLS[run.run_key]


@typechecked
def run_attempt(run: RunContext, team: Team, team_id: int, eval_index: int,
                resource_id: int) -> EvalResult:
    """One evaluation attempt, in this process.

    ``resource_id`` pins every model call of the attempt to one Azure resource (``R{n}_``),
    so the EA's allocator can keep concurrent attempts off each other's deployments. The
    Evaluator is built per attempt: it is cheap, and it holds no state between calls.
    """
    graph_builder = TeamGraphBuilder(run.factories, _resources(run), run.task,
                                     run.model_config, resource=resource_id,
                                     build_model=run.build_model)
    evaluator = Evaluator(task=run.task, catalog=run.catalog, model_config=run.model_config,
                          cv_folds=run.cv_folds, seed=run.seed, run_dir=run.run_dir,
                          build_graph=graph_builder)
    return evaluator.evaluate(team, team_id=team_id, eval_index=eval_index)


@ray.remote(num_cpus=1)
def evaluate_remote(run: RunContext, team: Team, team_id: int, eval_index: int,
                    resource_id: int) -> EvalResult:
    """``run_attempt`` as a Ray task: one attempt on one CPU of a worker process."""
    return run_attempt(run, team, team_id, eval_index, resource_id)
