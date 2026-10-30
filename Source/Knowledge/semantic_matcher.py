"""The semantic matcher: maps a concept an agent writes to the catalog features it means.

Agents never write a ``feature_name`` themselves; they describe a concept ("high blood
pressure") and copy a name the matcher returns ("I10 | HTN (hypertension)"). How a concept
is matched is up to the user: subclass ``SemanticMatcher``, implement ``match`` and hand an
instance to the EA.

GENASYS owns everything around ``match``: the ``Semantic_Matcher`` tool the agents call (its
name, description and output are what the prompts document), and the check that every
match is a catalog feature with the catalog's own ``feature_type`` and a finite score. A
matcher that breaks that contract raises ``MatcherContractError``, which stops the run: it
is a bug in the matcher, not something a team did, so it must not be scored as a team
failure.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from langchain_core.tools import BaseTool, tool
from typeguard import typechecked

from .catalog import CatalogEntry

# The tool's name, as every prompt calls it.
SEMANTIC_MATCHER_TOOL = "Semantic_Matcher"


@dataclass(frozen=True)
class Match:
    """One catalog feature a concept matched; higher ``similarity_score`` is closer."""

    feature_name: str
    feature_type: str
    similarity_score: float


class SemanticMatcher(ABC):
    """Maps a free-text concept to the catalog features it most likely means.

    Load models and embeddings in ``__init__`` (no lazy globals), so a broken setup fails
    when the matcher is built, before any team runs. ``match`` should be deterministic,
    so a team's run can be reproduced.
    """

    @abstractmethod
    def match(self, concept: str) -> Sequence[Match]:
        """The catalog features ``concept`` matches, best first; empty if none is close
        enough. ``concept`` is non-empty and stripped."""


class MatcherContractError(RuntimeError):
    """A ``SemanticMatcher`` returned something the run cannot use; stops the run."""


@typechecked
def check_matches(matches: Any, concept: str, catalog: Mapping[str, CatalogEntry],
                  matcher: SemanticMatcher) -> List[Match]:
    """``matches`` as a list, or ``MatcherContractError`` naming the first problem."""
    who = f"{type(matcher).__name__}.match({concept!r})"
    if not isinstance(matches, (list, tuple)):
        raise MatcherContractError(f"{who} returned {type(matches).__name__}, not a list "
                                   f"of Match")
    for m in matches:
        if not isinstance(m, Match):
            raise MatcherContractError(f"{who} returned a {type(m).__name__}, not a Match: "
                                       f"{m!r}")
        if m.feature_name not in catalog:
            raise MatcherContractError(f"{who} returned '{m.feature_name}', which is not in "
                                       f"the run's feature catalog; is the matcher built on "
                                       f"the same catalog as the EA?")
        expected = catalog[m.feature_name].feature_type
        if m.feature_type != expected:
            raise MatcherContractError(f"{who} returned '{m.feature_name}' as "
                                       f"'{m.feature_type}'; the catalog says '{expected}'")
        if (isinstance(m.similarity_score, bool)
                or not isinstance(m.similarity_score, (int, float))
                or not math.isfinite(m.similarity_score)):
            raise MatcherContractError(f"{who} returned '{m.feature_name}' with similarity "
                                       f"score {m.similarity_score!r}; it must be a finite "
                                       f"number")
    return list(matches)


@typechecked
def semantic_matcher_tool(matcher: SemanticMatcher,
                          catalog: Mapping[str, CatalogEntry]) -> BaseTool:
    """The ``Semantic_Matcher`` tool, backed by ``matcher`` and checked against ``catalog``.

    Returns ``{"candidates": [(feature_name, feature_type, similarity_score), ...]}``, best
    first, as the prompts document. An empty concept is answered, not passed on.
    """

    @tool(SEMANTIC_MATCHER_TOOL, description=(
        "Map one clinical concept to dataset features. concept: a plain clinical "
        "description (e.g. 'type 2 diabetes') or a raw ICD-10 code (e.g. 'E11.9'). Returns "
        "candidates, best first, each (feature_name, feature_type, similarity_score); an "
        "empty list means nothing matched closely enough. Copy feature_name verbatim."))
    def Semantic_Matcher(concept: str) -> Dict[str, Any]:
        concept = concept.strip()
        if not concept:
            return {"candidates": [], "note": "concept is empty; describe the clinical "
                                              "concept you want to match."}
        matches = check_matches(matcher.match(concept), concept, catalog, matcher)
        return {"candidates": [(m.feature_name, m.feature_type, m.similarity_score)
                               for m in matches]}

    return Semantic_Matcher
