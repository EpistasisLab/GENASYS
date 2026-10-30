"""The run-wide data the agent tools are built on.

The EA loads these once per run and hands the same object to every factory's
``tools()``, so no tool loads data on first use or keeps it in a global.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd
from typeguard import typechecked

from ...Knowledge.catalog import CatalogEntry
from ...Knowledge.knowledge_graph import KnowledgeGraph
from ...Knowledge.retriever import Retriever
from ...Knowledge.semantic_matcher import SemanticMatcher


@typechecked
@dataclass(frozen=True)
class ToolResources:
    # feature_name -> entry: the features an agent may stage, and the authority on
    # each one's feature_type.
    catalog: Mapping[str, CatalogEntry]
    # The user's matcher behind the Semantic_Matcher tool; its matches must be in catalog.
    matcher: SemanticMatcher
    # The rows the statistics tools analyse: the dev split, which the CV folds are cut
    # from. Never the holdout.
    dataset: pd.DataFrame
    # Its outcome column; encoded alongside the features, never analysed.
    target_column: str
    # The user's retriever behind the retrieve_context tool; None when no role searches
    # the literature.
    retriever: Optional[Retriever] = None
    # The user's knowledge graph behind the kg_* tools; None when no role searches it.
    kg: Optional[KnowledgeGraph] = None

    def __post_init__(self) -> None:
        assert self.catalog, "ToolResources needs a non-empty feature catalog."
        assert len(self.dataset), "ToolResources needs a non-empty dataset."
        assert self.target_column in self.dataset.columns, (
            f"target column '{self.target_column}' is not in the dataset")
        # @typechecked does not check dataclass fields, so a duck-typed matcher would pass.
        assert isinstance(self.matcher, SemanticMatcher), (
            f"the matcher must subclass SemanticMatcher, not {type(self.matcher).__name__}")
        assert self.retriever is None or isinstance(self.retriever, Retriever), (
            f"the retriever must subclass Retriever, not {type(self.retriever).__name__}")
        assert self.kg is None or isinstance(self.kg, KnowledgeGraph), (
            f"the knowledge graph must subclass KnowledgeGraph, not {type(self.kg).__name__}")
