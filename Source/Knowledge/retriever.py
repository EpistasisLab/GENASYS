"""The literature retriever: finds the corpus passages most relevant to an agent's query.

The RAG agent (and the single agent) ground their concepts and reviews in a curated
literature corpus. How the corpus is stored, embedded and searched is up to the user:
subclass ``Retriever``, implement ``retrieve`` and hand an instance to the EA.

GENASYS owns everything around ``retrieve``: the ``retrieve_context`` tool the agents call
(its name, arguments and JSON output are what the prompts document), the bounds on ``k``,
and the check that every answer is at most ``k`` passages, best first, each with a finite
score. What the score measures is the retriever's to define and the prompts' to explain;
GENASYS only requires that higher means more relevant. A retriever that breaks the contract
raises ``RetrieverContractError``, which stops the run: it is a bug in the retriever, not
something a team did, so it must not be scored as a team failure.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from langchain_core.tools import BaseTool, tool
from typeguard import typechecked

# The tool's name, as every prompt calls it.
RETRIEVE_CONTEXT_TOOL = "retrieve_context"
# The most passages one call may ask for; more would flood the agent's context.
MAX_K = 20
DIGITS = 4


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk of the corpus; higher ``score`` is more relevant."""

    title: str
    # Where the chunk came from (e.g. the PDF's file name), so a reader can trace it.
    source: str
    content: str
    score: float


class Retriever(ABC):
    """Finds the corpus passages most relevant to a free-text query.

    Open the corpus, load models and check every external service in ``__init__`` (no
    lazy globals), so a broken setup fails when the retriever is built, before any team
    runs. ``retrieve`` should be deterministic, so a team's run can be reproduced.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int) -> Sequence[Passage]:
        """At most ``k`` passages for ``query``, most relevant first. ``query`` is
        non-empty and stripped; ``1 <= k <= MAX_K``."""


class RetrieverContractError(RuntimeError):
    """A ``Retriever`` returned something the run cannot use; stops the run."""


@typechecked
def check_passages(passages: Any, query: str, k: int, retriever: Retriever) -> List[Passage]:
    """``passages`` as a list, or ``RetrieverContractError`` naming the first problem."""
    who = f"{type(retriever).__name__}.retrieve({query!r}, k={k})"
    if not isinstance(passages, (list, tuple)):
        raise RetrieverContractError(f"{who} returned {type(passages).__name__}, not a "
                                     f"list of Passage")
    if len(passages) > k:
        raise RetrieverContractError(f"{who} returned {len(passages)} passages, more than k")
    for p in passages:
        if not isinstance(p, Passage):
            raise RetrieverContractError(f"{who} returned a {type(p).__name__}, not a "
                                         f"Passage: {p!r}")
        for name in ("title", "source", "content"):
            if not isinstance(getattr(p, name), str):
                raise RetrieverContractError(f"{who} returned a passage whose {name} is "
                                             f"{getattr(p, name)!r}, not a string")
        if (isinstance(p.score, bool) or not isinstance(p.score, (int, float))
                or not math.isfinite(p.score)):
            raise RetrieverContractError(f"{who} returned a passage from '{p.source}' with "
                                         f"score {p.score!r}; it must be a finite number")
    scores = [p.score for p in passages]
    if scores != sorted(scores, reverse=True):
        raise RetrieverContractError(f"{who} returned passages out of order (scores "
                                     f"{scores}); the most relevant must come first")
    return list(passages)


@typechecked
def retrieve_context_tool(retriever: Retriever) -> BaseTool:
    """The ``retrieve_context`` tool, backed by ``retriever``.

    Returns ``{"passages": [{"rank", "title", "source", "score", "content"}, ...]}``, most
    relevant first, as the prompts document. An empty query or a ``k`` outside
    ``1..MAX_K`` is answered with a note, not passed on.
    """

    @tool(RETRIEVE_CONTEXT_TOOL, description=(
        "Search the curated literature corpus. query: a natural-language description of "
        f"the information you need. k: how many passages to return, 1 to {MAX_K}. Returns "
        "{\"passages\": [{rank, title, source, score, content}, ...]}, most relevant first "
        "(rank 1); a higher score is more relevant."))
    def retrieve_context(query: str, k: int) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            return {"passages": [], "note": "query is empty; describe the information "
                                            "you want from the literature."}
        if not 1 <= k <= MAX_K:
            return {"passages": [], "note": f"k must be between 1 and {MAX_K}; got {k}."}
        passages = check_passages(retriever.retrieve(query, k), query, k, retriever)
        return {"passages": [{"rank": i + 1, "title": p.title, "source": p.source,
                              "score": round(float(p.score), DIGITS), "content": p.content}
                             for i, p in enumerate(passages)]}

    return retrieve_context
