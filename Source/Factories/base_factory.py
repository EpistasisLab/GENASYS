"""Agent factories: everything a run needs to know about one team role.

A factory owns its role's wiring (role name, handoff tool, review fields, which
feature-state tools it gets), the description the Supervisor sees in its roster,
and the role's prompt templates.
Every template is read from a file the caller names; there are no built-in
prompts, so a factory cannot silently run with text nobody chose. The examples in
``Default_Prompts/`` are ordinary files to pass in or copy.

A template file is plain UTF-8 text of any extension (``.md``, ``.txt``, ...); it
reaches the model verbatim. Templates are ``str.format`` strings: a placeholder
such as ``{target_variable}`` is filled when the agent is built, and a literal
brace must be doubled (``{{`` / ``}}``), as in the JSON examples of the shipped
prompts. Every template is checked when its factory is constructed: a missing
file, a missing required placeholder, or an unknown one (usually a typo or an
unescaped brace) fails the run before any agent starts.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from langchain_core.tools import BaseTool
from typeguard import typechecked

from ..Agents.Tools.dataset_tools import dataset_schema_tool
from ..Agents.Tools.feature_state import (decision_tools, inspector_tools,
                                         reject_concept_tool, review_tool,
                                         staging_tool)
from ..Agents.Tools.stats_tools import stats_tools
from ..Agents.Tools.tool_resources import ToolResources
from ..Knowledge.knowledge_graph import kg_tools
from ..Knowledge.retriever import retrieve_context_tool
from ..Knowledge.semantic_matcher import semantic_matcher_tool

# Placeholders any agent's templates may use. Their values are rendered per run
# (target, file) and per team (the roster-dependent blocks).
COMMON_PLACEHOLDERS: FrozenSet[str] = frozenset({
    "target_variable", "file_name",
    "team_members", "agent_roster", "agent_call_tools",
    "origin_roles", "pending_review_fields",
})

# What every worker's system prompt has to carry: the target, its teammates, and
# the pending_features schema it reads and writes.
WORKER_SYSTEM_PLACEHOLDERS: FrozenSet[str] = frozenset({
    "target_variable", "team_members", "origin_roles", "pending_review_fields"})


def read_prompt_file(path: Union[str, Path], who: str) -> str:
    """The text of the template file at ``path``; fails if there is none.

    The extension is irrelevant: any UTF-8 text file works. A binary document
    (.docx, .pdf) fails here with a message rather than as a decode error.
    """
    assert str(path).strip(), f"{who}: no prompt file was given."
    file = Path(path).expanduser()
    assert file.is_file(), f"{who}: prompt file not found: {file}"
    try:
        return file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{who}: {file} is not a UTF-8 text file; save the prompt "
                         f"as plain text (.md or .txt).") from exc


def template_placeholders(template: str) -> FrozenSet[str]:
    """The field names a ``str.format`` template references.

    ``{a.b}`` and ``{a[0]}`` count as ``a``; a bare ``{}`` counts as ``''``.
    Raises ValueError on an unbalanced brace.
    """
    names = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            names.add(field.split(".", 1)[0].split("[", 1)[0])
    return frozenset(names)


@typechecked
class AgentFactory:
    """Base factory. Subclasses set the class-level wiring below.

    The wiring (``role_name``, ``call_tool_name``, ``review_fields`` and the tool
    flags) is fixed per class because tools and the handoff machinery depend on it. The templates
    (always from caller-supplied files) and the description are per instance.
    """

    role_name: ClassVar[str]
    # Short name used in the Supervisor's handoff-tool list, e.g. "RAG".
    display_name: ClassVar[str]
    # Handoff tool the Supervisor calls to reach this role. Empty for the
    # Supervisor itself, which is never handed off to.
    call_tool_name: ClassVar[str] = ""
    # The pending_features review fields this role writes: field name -> what goes in
    # it, as the prompts describe it. A worker has one, the Supervisor of a team none
    # (it only manages the team), the single agent one per perspective it covers.
    # Each field is written only by the role that declares it.
    review_fields: ClassVar[Dict[str, str]]
    default_description: ClassVar[str]

    # Which feature-state tools this role gets on top of its review tools and the
    # inspectors (see Agents/Tools/feature_state.py). Off unless a role turns it on.
    # Stages candidates with add_pending_feature.
    stages: ClassVar[bool] = False
    # Drops a concept that was never staged with add_rejected_concept.
    rejects_concepts: ClassVar[bool] = False
    # Accepts and rejects pending features and finalizes the set.
    decides: ClassVar[bool] = False
    # Maps concepts to catalog features with Semantic_Matcher.
    matches_concepts: ClassVar[bool] = False
    # Analyses features on the dev split with the statistics tools (Agents/Tools/stats_tools.py).
    analyzes_data: ClassVar[bool] = False
    # Searches the literature with the user's Retriever (Knowledge/retriever.py).
    retrieves: ClassVar[bool] = False
    # Searches the user's KnowledgeGraph (Knowledge/knowledge_graph.py).
    searches_kg: ClassVar[bool] = False
    # Reads the dev split's columns with print_dataset_schema (Agents/Tools/dataset_tools.py).
    describes_data: ClassVar[bool] = False

    # The templates this role has. Every one must be supplied as a file.
    prompt_keys: ClassVar[Tuple[str, ...]] = ("system", "user")
    # template key -> placeholders that template must contain.
    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {}
    # Placeholders only this role may use, on top of COMMON_PLACEHOLDERS.
    role_placeholders: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self,
                 system_prompt_path: Union[str, Path],
                 user_prompt_path: Union[str, Path],
                 description: Optional[str] = None,
                 **extra_prompt_paths: Union[str, Path]) -> None:
        """
        - system_prompt_path / user_prompt_path:
            Files holding the system and user templates. Required, and each must
            exist.
        - description:
            One line shown to teammates in their roster. ``None`` keeps the default.
        - extra_prompt_paths:
            Files for roles with further templates;
            keys must match ``prompt_keys``.
        """
        paths = {"system": system_prompt_path, "user": user_prompt_path,
                 **extra_prompt_paths}
        name = type(self).__name__
        assert set(paths) == set(self.prompt_keys), (
            f"{name} needs prompt files for exactly {sorted(self.prompt_keys)}, "
            f"got {sorted(paths)}.")

        self.description = description or self.default_description
        # Kept so a run's output can record which prompt files it used.
        self.prompt_paths: Dict[str, str] = {}
        self.templates: Dict[str, str] = {}
        for key in self.prompt_keys:
            template = read_prompt_file(paths[key], f"{name} '{key}' prompt")
            self._check_template(key, template)
            self.prompt_paths[key] = str(Path(paths[key]).expanduser())
            self.templates[key] = template

    def preflight(self) -> None:
        """Check the external services this role needs are reachable.

        The EA calls this once at run start, before any LLM spend, on every
        factory some allowed team is built from. Nothing to check by default;
        a role backed by its own server overrides it and
        raises with an actionable message when the server is down.
        """
        return

    def tools(self, team: Sequence[AgentFactory], resources: ToolResources) -> List[BaseTool]:
        """This role's tools on ``team`` (``self`` included).

        Staging validates ``suggested_by`` against the team's roles and seeds every
        review field of the team, so the tools depend on who else is on it.
        """
        team = _team_order(self, team)
        tools: List[BaseTool] = []
        if self.matches_concepts:
            tools.append(semantic_matcher_tool(resources.matcher, resources.catalog))
        if self.retrieves:
            assert resources.retriever is not None, (
                f"{type(self).__name__} searches the literature, but the run has no retriever")
            tools.append(retrieve_context_tool(resources.retriever))
        if self.searches_kg:
            assert resources.kg is not None, (
                f"{type(self).__name__} searches the knowledge graph, but the run has none")
            tools += kg_tools(resources.kg, resources.catalog)
        if self.stages:
            tools.append(staging_tool(self.role_name, [f.role_name for f in team],
                                      list(team_review_fields(team)), resources.catalog))
        tools += [review_tool(field) for field in self.review_fields]
        if self.analyzes_data:
            tools += stats_tools(resources)
        if self.rejects_concepts:
            tools.append(reject_concept_tool(self.role_name))
        if self.decides:
            tools += decision_tools()
        if self.describes_data:
            tools.append(dataset_schema_tool(resources))
        tools += inspector_tools()

        names = [t.name for t in tools]
        assert len(set(names)) == len(names), \
            f"{type(self).__name__} got two tools with the same name: {names}"
        return tools

    def allowed_placeholders(self) -> FrozenSet[str]:
        return COMMON_PLACEHOLDERS | self.role_placeholders

    def _check_template(self, key: str, template: str) -> None:
        """Fail if ``template`` lacks a required placeholder or uses an unknown one."""
        who = f"{type(self).__name__} '{key}' prompt"
        assert template.strip(), f"{who} is empty."
        try:
            found = template_placeholders(template)
        except ValueError as exc:
            raise ValueError(f"{who} has an unbalanced brace ({exc}). Write a literal "
                             f"brace as '{{{{' or '}}}}'.") from exc

        missing = self.required_placeholders.get(key, frozenset()) - found
        assert not missing, (
            f"{who} is missing required placeholder(s) "
            f"{', '.join('{' + m + '}' for m in sorted(missing))}.")

        unknown = found - self.allowed_placeholders()
        assert not unknown, (
            f"{who} uses unknown placeholder(s) "
            f"{', '.join('{' + u + '}' for u in sorted(unknown))}. Allowed: "
            f"{', '.join(sorted(self.allowed_placeholders()))}. If a brace is meant "
            f"literally, double it.")

    def render(self, key: str, subs: Dict[str, str]) -> str:
        """Fill template ``key`` from ``subs``, which must cover its placeholders."""
        template = self.templates[key]
        missing = template_placeholders(template) - set(subs)
        assert not missing, (
            f"no value for {sorted(missing)} when rendering "
            f"{type(self).__name__} '{key}' prompt.")
        return template.format(**subs)


def team_placeholders(me: AgentFactory, team: Sequence[AgentFactory]) -> Dict[str, str]:
    """The roster-dependent placeholder values for ``me`` on this team.

    ``team`` is the whole team, ``me`` included, one factory per role (a team's
    roster is keyed by role, so it cannot repeat one), and should be the same
    sequence for every member so each block reads the same way in every prompt.
    Every block is built from the factories actually present, so a prompt never
    advertises an agent, handoff tool or review field that this team does not have.
    Each block is a markdown list, one ``- `` line per item, as the prompts' lists are.
    """
    everyone = _team_order(me, team)
    mates = [f for f in everyone if f.role_name != me.role_name]
    # Only roles with a handoff tool can be delegated to (never the Supervisor).
    callable_mates = [f for f in mates if f.call_tool_name]

    return {
        "team_members": "\n".join(f"- {f.role_name}: {f.description}" for f in mates)
                        or "(none)",
        "agent_roster": "\n".join(f"- {f.call_tool_name}: {f.description}"
                                  for f in callable_mates)
                        or "(no worker agents in this pipeline)",
        "agent_call_tools": "\n".join(
            f"- {f.call_tool_name}(): invoke the {f.display_name} agent for assistance."
            for f in callable_mates)
                            or "(no worker agents available to call)",
        # The roles a suggested_by value may name: only agents on this team, or
        # the Supervisor would credit an agent that was never connected.
        "origin_roles": ", ".join(f.role_name for f in everyone),
        "pending_review_fields": "\n".join(
            f"- {field} (list of strings): {meaning}, newest last — a new review is "
            f"appended, never overwritten"
            for field, meaning in team_review_fields(everyone).items())
                                 or "(no review fields on this team)",
    }


def _team_order(me: AgentFactory, team: Sequence[AgentFactory]) -> List[AgentFactory]:
    """``team`` with the Supervisor first and everyone else in the team's order.

    Fails if the team repeats a role or does not include ``me``.
    """
    roles = [f.role_name for f in team]
    assert len(set(roles)) == len(roles), f"team repeats a role: {roles}"
    assert me in team, f"{type(me).__name__} is not on the team {roles}"
    return sorted(team, key=lambda f: f.role_name != "supervisor")


def team_review_fields(team: Sequence[AgentFactory]) -> Dict[str, str]:
    """Every review field on this team (field -> what goes in it), in team order.

    A pending feature carries exactly these fields; a field of an agent the team does
    not have never appears. Raises if two roles declare the same field, since each
    field must have one writer.
    """
    fields: Dict[str, str] = {}
    for factory in team:
        for field, meaning in factory.review_fields.items():
            assert field.endswith("_review"), \
                f"{type(factory).__name__}: review field '{field}' must end in '_review'"
            assert field not in fields, \
                f"review field '{field}' is declared by two roles on the team"
            fields[field] = meaning
    return fields
