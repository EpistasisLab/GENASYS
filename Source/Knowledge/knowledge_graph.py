"""The knowledge graph: biomedical structure the KG agent (and the single agent) traverse.

A graph node is worth recommending only if it maps to a feature the cohort records, so a
knowledge graph here is two things: the graph itself and a mapping from its clinical
nodes to catalog features. How both are stored, searched and scored is up to the user:
subclass ``KnowledgeGraph``, implement its methods and hand an instance to the EA.

GENASYS owns everything around those methods: the four ``kg_*`` tools the agents call (their
names, arguments and JSON output are what the prompts document), the bounds on their
arguments, the ``{kg_schema_summary}`` the prompts show, and the check that every answer
is well formed. A node's features are named only; their ``feature_type`` and
``prevalence`` come from the run's catalog, so every mapped feature must be in it. A graph
that breaks the contract raises ``KGContractError``, which stops the run: it is a bug in
the graph, not something a team did, so it must not be scored as a team failure. An
agent's bad argument (an unknown label, a ``k`` out of range) is answered with a note.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from langchain_core.tools import BaseTool, tool
from typeguard import typechecked

from .catalog import CatalogEntry

# The tools' names, as every prompt calls them.
KG_FIND_NODE_TOOL = "kg_find_node"
KG_NEIGHBORS_TOOL = "kg_neighbors"
KG_LINK_PREDICTION_TOOL = "kg_link_prediction"
KG_FEATURE_LOOKUP_TOOL = "kg_feature_lookup"
KG_TOOLS = (KG_FIND_NODE_TOOL, KG_NEIGHBORS_TOOL, KG_LINK_PREDICTION_TOOL,
            KG_FEATURE_LOOKUP_TOOL)
# The most nodes one call may ask for; more would flood the agent's context.
MAX_K = 50
# The deepest walk: past 3 hops a seed reaches most of a gene-dense graph.
MAX_DEPTH = 3
# The most unmapped candidates a link prediction reports; context, not a second list.
MAX_UNMAPPED = 5
DIGITS = 4


@dataclass(frozen=True)
class FeatureLink:
    """A catalog feature a node maps to. ``similarity`` is the mapping's confidence,
    None when it is not a score (an exact code join, a hand-written override)."""

    feature_name: str
    similarity: Optional[float] = None


@dataclass(frozen=True)
class Node:
    """One graph node as an agent sees it. ``features`` are the catalog features it maps
    to, best first; empty when it can never be proposed (a gene, a pathway) or when the
    mapping did not confirm one."""

    name: str
    label: str
    degree: int
    features: Tuple[FeatureLink, ...] = ()


@dataclass(frozen=True)
class Reached:
    """A node a walk reached, ``hops`` from the seed; ``path`` is the evidence, e.g.
    'DISEASE -[ASSOCIATES]-> TNF -[ASSOCIATES]-> tuberculosis'."""

    node: Node
    hops: int
    path: str


@dataclass(frozen=True)
class Neighborhood:
    """A walk from ``seed``: how many nodes it reached, and the mapped ones reported."""

    seed: Node
    n_reached: int
    reached: Tuple[Reached, ...]


@dataclass(frozen=True)
class Candidate:
    """A node scored against a seed; higher ``score`` is a likelier link. ``shared`` is
    the number of neighbours the two share, and ``example_shared`` names a few of them."""

    node: Node
    score: float
    shared: int
    example_shared: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Ranking:
    """A link prediction from ``seed``: ``candidates`` are mapped, ``unmapped`` are the
    best-scoring nodes with no feature; both best first."""

    seed: Node
    n_scored: int
    candidates: Tuple[Candidate, ...]
    unmapped: Tuple[Candidate, ...] = ()


@dataclass(frozen=True)
class KGSchema:
    """What the graph holds; shown to the agents and used to check their arguments."""

    n_nodes: int
    n_edges: int
    # node label -> number of nodes
    labels: Mapping[str, int]
    relationship_types: Tuple[str, ...]
    # The scoring heuristics ``link_prediction`` accepts; the first is the default.
    link_methods: Tuple[str, ...]


class KnowledgeGraph(ABC):
    """A biomedical graph whose clinical nodes map to catalog features.

    Load the graph and its mapping in ``__init__`` (no lazy globals), so a broken setup
    fails when the graph is built, before any team runs. Every method should be
    deterministic, so a team's run can be reproduced. A ``seed`` or ``name`` is free text
    an agent wrote: a node name or a catalog ``feature_name``; resolve either. Arguments
    reach these methods already checked: text non-empty and stripped, ``1 <= k <=
    MAX_K``, ``1 <= depth <= MAX_DEPTH``, every label, relationship type and method one
    ``schema()`` lists.
    """

    @abstractmethod
    def schema(self) -> KGSchema:
        """The graph's labels, relationship types and link-prediction methods."""

    @abstractmethod
    def find_nodes(self, query: str, k: int,
                   labels: Optional[Tuple[str, ...]]) -> Sequence[Node]:
        """At most ``k`` nodes matching ``query``, best first, restricted to ``labels``
        when given."""

    @abstractmethod
    def neighbors(self, seed: str, depth: int, k: int,
                  relationship_types: Optional[Tuple[str, ...]]) -> Optional[Neighborhood]:
        """A walk of up to ``depth`` hops from ``seed`` along ``relationship_types`` (all
        when None), reporting at most ``k`` mapped nodes; None if ``seed`` resolves to no
        node."""

    @abstractmethod
    def link_prediction(self, seed: str, method: str, k: int,
                        labels: Optional[Tuple[str, ...]]) -> Optional[Ranking]:
        """At most ``k`` mapped and ``MAX_UNMAPPED`` unmapped nodes scored against
        ``seed`` by ``method``, restricted to ``labels`` when given; None if ``seed``
        resolves to no node."""

    @abstractmethod
    def lookup(self, name: str) -> Optional[Node]:
        """The node ``name`` resolves to, or None."""


class KGContractError(RuntimeError):
    """A ``KnowledgeGraph`` returned something the run cannot use; stops the run."""


def _finite(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)


def _count(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, int) and x >= 0


def _names(items: Any) -> bool:
    return isinstance(items, tuple) and all(isinstance(s, str) and s for s in items)


@typechecked
def check_schema(schema: Any, graph: KnowledgeGraph) -> KGSchema:
    """``schema``, or ``KGContractError`` naming the first problem."""
    who = f"{type(graph).__name__}.schema()"
    if not isinstance(schema, KGSchema):
        raise KGContractError(f"{who} returned {type(schema).__name__}, not a KGSchema")
    if not (_count(schema.n_nodes) and _count(schema.n_edges) and schema.n_nodes > 0):
        raise KGContractError(f"{who} counts {schema.n_nodes!r} nodes and "
                              f"{schema.n_edges!r} edges; both must be counts, nodes > 0")
    if not (isinstance(schema.labels, Mapping) and schema.labels
            and all(isinstance(l, str) and l and _count(n)
                    for l, n in schema.labels.items())):
        raise KGContractError(f"{who} has labels {schema.labels!r}; they must map each "
                              f"label to its node count")
    for name in ("relationship_types", "link_methods"):
        if not (_names(getattr(schema, name)) and getattr(schema, name)):
            raise KGContractError(f"{who} has {name} {getattr(schema, name)!r}; it must be "
                                  f"a non-empty tuple of names")
    return schema


class _Checker:
    """Checks one call's answer against the schema and the catalog."""

    def __init__(self, who: str, schema: KGSchema, catalog: Mapping[str, CatalogEntry]):
        self.who, self.schema, self.catalog = who, schema, catalog

    def fail(self, problem: str) -> None:
        raise KGContractError(f"{self.who} returned {problem}")

    def node(self, node: Any, labels: Optional[Tuple[str, ...]] = None) -> None:
        if not isinstance(node, Node):
            self.fail(f"a {type(node).__name__}, not a Node: {node!r}")
        if not (isinstance(node.name, str) and node.name):
            self.fail(f"a node named {node.name!r}; it needs a name")
        if node.label not in self.schema.labels:
            self.fail(f"node '{node.name}' labelled {node.label!r}, which schema() does "
                      f"not list")
        if labels is not None and node.label not in labels:
            self.fail(f"node '{node.name}' labelled '{node.label}', outside the requested "
                      f"labels {list(labels)}")
        if not _count(node.degree):
            self.fail(f"node '{node.name}' with degree {node.degree!r}")
        if not isinstance(node.features, tuple):
            self.fail(f"node '{node.name}' whose features are a "
                      f"{type(node.features).__name__}, not a tuple of FeatureLink")
        seen = set()
        for f in node.features:
            if not isinstance(f, FeatureLink):
                self.fail(f"node '{node.name}' with a {type(f).__name__} feature, not a "
                          f"FeatureLink: {f!r}")
            if f.feature_name not in self.catalog:
                self.fail(f"node '{node.name}' mapped to '{f.feature_name}', which is not "
                          f"in the run's feature catalog; is the mapping built on the "
                          f"same catalog as the EA?")
            if f.feature_name in seen:
                self.fail(f"node '{node.name}' mapped to '{f.feature_name}' twice")
            seen.add(f.feature_name)
            if f.similarity is not None and not _finite(f.similarity):
                self.fail(f"node '{node.name}' mapped to '{f.feature_name}' with "
                          f"similarity {f.similarity!r}; it must be finite or None")

    def ordered(self, scores: List[float], what: str) -> None:
        if scores != sorted(scores, reverse=True):
            self.fail(f"{what} out of order (scores {scores}); the best must come first")

    def candidate(self, c: Any, labels: Optional[Tuple[str, ...]], mapped: bool) -> None:
        if not isinstance(c, Candidate):
            self.fail(f"a {type(c).__name__}, not a Candidate: {c!r}")
        self.node(c.node, labels)
        if bool(c.node.features) != mapped:
            self.fail(f"'{c.node.name}' among the {'' if mapped else 'un'}mapped "
                      f"candidates, but it has {len(c.node.features)} feature(s)")
        if not _finite(c.score):
            self.fail(f"'{c.node.name}' with score {c.score!r}; it must be a finite number")
        if not _count(c.shared):
            self.fail(f"'{c.node.name}' with {c.shared!r} shared neighbours")
        if not (isinstance(c.example_shared, tuple)
                and all(isinstance(s, str) for s in c.example_shared)):
            self.fail(f"'{c.node.name}' with example_shared {c.example_shared!r}; it must "
                      f"be a tuple of names")


@typechecked
def check_nodes(nodes: Any, checker: _Checker, k: int,
                labels: Optional[Tuple[str, ...]]) -> List[Node]:
    if not isinstance(nodes, (list, tuple)):
        checker.fail(f"{type(nodes).__name__}, not a list of Node")
    if len(nodes) > k:
        checker.fail(f"{len(nodes)} nodes, more than k")
    for node in nodes:
        checker.node(node, labels)
    return list(nodes)


@typechecked
def check_neighborhood(hood: Any, checker: _Checker, depth: int, k: int
                       ) -> Optional[Neighborhood]:
    if hood is None:
        return None
    if not isinstance(hood, Neighborhood):
        checker.fail(f"{type(hood).__name__}, not a Neighborhood or None")
    checker.node(hood.seed)
    if not isinstance(hood.reached, tuple):
        checker.fail(f"reached as a {type(hood.reached).__name__}, not a tuple")
    if len(hood.reached) > k:
        checker.fail(f"{len(hood.reached)} reached nodes, more than k")
    if not (_count(hood.n_reached) and hood.n_reached >= len(hood.reached)):
        checker.fail(f"n_reached={hood.n_reached!r} with {len(hood.reached)} reported")
    for r in hood.reached:
        if not isinstance(r, Reached):
            checker.fail(f"a {type(r).__name__}, not a Reached: {r!r}")
        checker.node(r.node)
        if not r.node.features:
            checker.fail(f"'{r.node.name}' as reached, but it maps to no feature; report "
                         f"only mapped nodes")
        if isinstance(r.hops, bool) or not isinstance(r.hops, int) or not 1 <= r.hops <= depth:
            checker.fail(f"'{r.node.name}' at {r.hops!r} hops; it must be 1 to {depth}")
        if not (isinstance(r.path, str) and r.path):
            checker.fail(f"'{r.node.name}' with no path")
    return hood


@typechecked
def check_ranking(ranking: Any, checker: _Checker, k: int,
                  labels: Optional[Tuple[str, ...]]) -> Optional[Ranking]:
    if ranking is None:
        return None
    if not isinstance(ranking, Ranking):
        checker.fail(f"{type(ranking).__name__}, not a Ranking or None")
    checker.node(ranking.seed)
    for field, limit, mapped in (("candidates", k, True), ("unmapped", MAX_UNMAPPED, False)):
        items = getattr(ranking, field)
        if not isinstance(items, tuple):
            checker.fail(f"{field} as a {type(items).__name__}, not a tuple")
        if len(items) > limit:
            checker.fail(f"{len(items)} {field}, more than {limit}")
        for c in items:
            checker.candidate(c, labels, mapped)
        checker.ordered([c.score for c in items], field)
    n = len(ranking.candidates) + len(ranking.unmapped)
    if not (_count(ranking.n_scored) and ranking.n_scored >= n):
        checker.fail(f"n_scored={ranking.n_scored!r} with {n} candidates reported")
    return ranking


@typechecked
def check_kg(graph: KnowledgeGraph, catalog: Mapping[str, CatalogEntry]) -> KGSchema:
    """The graph's schema, after a probe: ``lookup`` and ``find_nodes`` answer the first
    catalog entry's ``match_text`` within the contract, or ``KGContractError``. Called
    once at run start, so a broken graph fails before any team runs."""
    schema = check_schema(graph.schema(), graph)
    query = next(iter(catalog.values())).match_text.strip()
    gname = type(graph).__name__
    node = graph.lookup(query)
    if node is not None:
        _Checker(f"{gname}.lookup({query!r})", schema, catalog).node(node)
    check_nodes(graph.find_nodes(query, 1, None),
                _Checker(f"{gname}.find_nodes({query!r}, k=1, labels=None)", schema, catalog),
                1, None)
    return schema


@typechecked
def kg_schema_summary(schema: KGSchema) -> str:
    """The ``{kg_schema_summary}`` placeholder: what the graph holds, in prompt text."""
    labels = ", ".join(f"{l} ({n})" for l, n in
                       sorted(schema.labels.items(), key=lambda kv: (-kv[1], kv[0])))
    return (f"{schema.n_nodes} nodes and {schema.n_edges} edges.\n"
            f"Node labels (node count): {labels}\n"
            f"Relationship types: {', '.join(schema.relationship_types)}\n"
            f"Link-prediction methods: {', '.join(schema.link_methods)} "
            f"(default {schema.link_methods[0]})")


def _node_json(node: Node, catalog: Mapping[str, CatalogEntry]) -> Dict[str, Any]:
    """A node as the prompts document it: identity, then its features or
    ``"proposable": false``."""

    def feature(f: FeatureLink) -> Dict[str, Any]:
        entry = catalog[f.feature_name]
        out = {"feature_name": f.feature_name, "feature_type": entry.feature_type,
               "prevalence": round(float(entry.prevalence), DIGITS)}
        if f.similarity is not None:
            out["mapping_similarity"] = round(float(f.similarity), DIGITS)
        return out

    out: Dict[str, Any] = {"node": node.name, "label": node.label, "degree": node.degree,
                           "proposable": bool(node.features)}
    if node.features:
        out.update(feature(node.features[0]))
        if len(node.features) > 1:
            out["also_maps_to"] = [feature(f) for f in node.features[1:]]
    return out


def _candidate_json(c: Candidate, catalog: Mapping[str, CatalogEntry]) -> Dict[str, Any]:
    return {**_node_json(c.node, catalog), "score": round(float(c.score), DIGITS),
            "shared_neighbors": c.shared, "example_shared": list(c.example_shared)}


@typechecked
def kg_tools(graph: KnowledgeGraph, catalog: Mapping[str, CatalogEntry]) -> List[BaseTool]:
    """The four ``kg_*`` tools, backed by ``graph`` and checked against ``catalog``.

    Each returns a JSON object, as the prompts document; a bad argument is answered with
    ``{"note": ...}`` and not passed on.
    """
    schema = check_schema(graph.schema(), graph)
    known_labels = sorted(schema.labels)
    known_rels = list(schema.relationship_types)
    methods = list(schema.link_methods)
    gname = type(graph).__name__

    def bad_text(name: str, value: str) -> Optional[Dict[str, Any]]:
        if not value:
            return {"note": f"{name} is empty; give a node name or a feature_name."}
        return None

    def bad_k(k: int) -> Optional[Dict[str, Any]]:
        if not 1 <= k <= MAX_K:
            return {"note": f"k must be between 1 and {MAX_K}; got {k}."}
        return None

    def bad_names(name: str, given: Optional[List[str]], known: List[str]
                  ) -> Optional[Dict[str, Any]]:
        unknown = sorted(set(given or ()) - set(known))
        if unknown:
            return {"note": f"{name} {unknown} are not in this graph; known: {known}."}
        return None

    def as_tuple(given: Optional[List[str]]) -> Optional[Tuple[str, ...]]:
        return tuple(dict.fromkeys(given)) if given else None

    def no_seed(seed: str) -> Dict[str, Any]:
        return {"note": f"no node in the graph matches '{seed}'; try {KG_FIND_NODE_TOOL} "
                        f"first."}

    @tool(KG_FIND_NODE_TOOL, description=(
        "Resolve a clinical term to knowledge-graph nodes; call it before a traversal. "
        f"query: a node name or a cohort feature_name. k: how many nodes, 1 to {MAX_K}. "
        "labels: optional node labels to restrict the search to. Returns {\"query\", "
        "\"matches\": [node, ...]}, best first."))
    def kg_find_node(query: str, k: int = 10,
                     labels: Optional[List[str]] = None) -> Dict[str, Any]:
        query = query.strip()
        note = (bad_text("query", query) or bad_k(k)
                or bad_names("labels", labels, known_labels))
        if note:
            return note
        wanted = as_tuple(labels)
        checker = _Checker(f"{gname}.find_nodes({query!r}, k={k}, labels={wanted})",
                           schema, catalog)
        nodes = check_nodes(graph.find_nodes(query, k, wanted), checker, k, wanted)
        return {"query": query, "matches": [_node_json(n, catalog) for n in nodes]}

    @tool(KG_NEIGHBORS_TOOL, description=(
        "Walk outward from a seed node and return the mapped nodes reached, each with the "
        "path that connects it. seed: a node name or a cohort feature_name. depth: hops, "
        f"1 to {MAX_DEPTH}. k: how many nodes to report, 1 to {MAX_K}. "
        "relationship_types: optional edge types to walk. Returns {\"seed\", "
        "\"seed_label\", \"depth\", \"n_nodes_reached\", \"results\": [node + hops + "
        "path, ...]}."))
    def kg_neighbors(seed: str, depth: int = 2, k: int = 25,
                     relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        seed = seed.strip()
        note = (bad_text("seed", seed) or bad_k(k)
                or bad_names("relationship_types", relationship_types, known_rels))
        if not note and not 1 <= depth <= MAX_DEPTH:
            note = {"note": f"depth must be between 1 and {MAX_DEPTH}; got {depth}."}
        if note:
            return note
        rels = as_tuple(relationship_types)
        checker = _Checker(f"{gname}.neighbors({seed!r}, depth={depth}, k={k}, "
                           f"relationship_types={rels})", schema, catalog)
        hood = check_neighborhood(graph.neighbors(seed, depth, k, rels), checker, depth, k)
        if hood is None:
            return no_seed(seed)
        return {"seed": hood.seed.name, "seed_label": hood.seed.label, "depth": depth,
                "n_nodes_reached": hood.n_reached,
                "results": [{**_node_json(r.node, catalog), "hops": r.hops, "path": r.path}
                            for r in hood.reached]}

    @tool(KG_LINK_PREDICTION_TOOL, description=(
        "Rank the mapped nodes most likely to be linked to a seed node, by shared "
        "neighbours. seed: a node name or a cohort feature_name. method: one of "
        f"{', '.join(methods)} (default {methods[0]}). k: how many, 1 to {MAX_K}. labels: "
        "optional node labels to restrict candidates to. Returns {\"seed\", \"method\", "
        "\"n_candidates_scored\", \"results\": [node + score + shared_neighbors + "
        "example_shared, ...], \"unmapped_high_rank\": [...]}, best first."))
    def kg_link_prediction(seed: str, method: str = methods[0], k: int = 10,
                           labels: Optional[List[str]] = None) -> Dict[str, Any]:
        seed, method = seed.strip(), method.strip()
        note = (bad_text("seed", seed) or bad_k(k)
                or bad_names("labels", labels, known_labels))
        if not note and method not in methods:
            note = {"note": f"method '{method}' is not one of {methods}."}
        if note:
            return note
        wanted = as_tuple(labels)
        checker = _Checker(f"{gname}.link_prediction({seed!r}, {method!r}, k={k}, "
                           f"labels={wanted})", schema, catalog)
        ranking = check_ranking(graph.link_prediction(seed, method, k, wanted), checker, k,
                                wanted)
        if ranking is None:
            return no_seed(seed)
        return {"seed": ranking.seed.name, "method": method,
                "n_candidates_scored": ranking.n_scored,
                "results": [_candidate_json(c, catalog) for c in ranking.candidates],
                "unmapped_high_rank": [_candidate_json(c, catalog) for c in ranking.unmapped]}

    @tool(KG_FEATURE_LOOKUP_TOOL, description=(
        "Check whether one knowledge-graph node maps to cohort features. name: a node "
        "name or a cohort feature_name. Returns the node with \"found\": true, or "
        "{\"node\", \"found\": false, \"proposable\": false} when nothing resolves."))
    def kg_feature_lookup(name: str) -> Dict[str, Any]:
        name = name.strip()
        note = bad_text("name", name)
        if note:
            return note
        checker = _Checker(f"{gname}.lookup({name!r})", schema, catalog)
        node = graph.lookup(name)
        if node is None:
            return {"node": name, "found": False, "proposable": False}
        checker.node(node)
        return {**_node_json(node, catalog), "found": True}

    return [kg_find_node, kg_neighbors, kg_link_prediction, kg_feature_lookup]
