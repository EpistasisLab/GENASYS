"""The run's topology rules, checked once and expanded into every allowed team.

A team is the set of role names on it; the Supervisor slot (``"supervisor"``) is
always present. The user-facing rules are a dict:

- ``required_workers``: workers every team keeps.
- ``available_workers``: workers a team may have. Defaults to every worker
  role the run has a factory for.
- ``restricted_teams``: exact compositions that are not allowed, e.g.
  ``({"supervisor", "statistician"},)``. Matching is exact: the pair is banned,
  a larger team containing it is not. ``"supervisor"`` may be left out of an
  entry; it is added.

Every allowed team is enumerated up front, so Reproduction can check a team by
lookup and fall back to a random allowed team, and the EA can check that it
has a factory for every role an allowed team may contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

from ..Utils.log import warn

SUPERVISOR = "supervisor"

_RULE_KEYS = frozenset({"required_workers", "available_workers", "restricted_teams"})


@dataclass(frozen=True)
class TopologyRules:
    required_workers: FrozenSet[str]
    available_workers: FrozenSet[str]
    restricted_teams: FrozenSet[FrozenSet[str]]
    # Every allowed composition, in a fixed order so random draws over it are
    # reproducible from the seed.
    allowed_teams: Tuple[FrozenSet[str], ...]

    def is_allowed(self, team: Iterable[str]) -> bool:
        return frozenset(team) in self.allowed_teams

    def single_agent_allowed(self) -> bool:
        """Can a team be the Supervisor slot alone (the single-agent baseline)?"""
        return self.is_allowed({SUPERVISOR})

    def roles_in_use(self) -> FrozenSet[str]:
        """Every role at least one allowed team contains."""
        return frozenset().union(*self.allowed_teams)


def _role_set(value: Any, what: str) -> FrozenSet[str]:
    assert isinstance(value, (set, frozenset, list, tuple)), (
        f"{what} must be a set of role names, got {type(value).__name__}: {value!r}")
    assert all(isinstance(v, str) for v in value), f"{what} must hold role names: {value!r}"
    return frozenset(value)


def build_topology_rules(rules: Optional[Dict[str, Any]],
                         worker_roles: Iterable[str]) -> TopologyRules:
    """Check ``rules`` against the worker roles the run has factories for, and
    enumerate the allowed teams. Fails on an unknown key, an unknown role, a
    required worker that is not available, or rules that allow no team at all.
    """
    rules = dict(rules or {})
    workers = frozenset(worker_roles)
    assert SUPERVISOR not in workers, "The Supervisor slot is not a worker role."

    unknown_keys = set(rules) - _RULE_KEYS
    assert not unknown_keys, (
        f"Unknown topology_rules key(s) {sorted(unknown_keys)}; expected "
        f"{sorted(_RULE_KEYS)}.")

    def check_workers(roles: FrozenSet[str], what: str) -> None:
        assert SUPERVISOR not in roles, (
            f"{what} lists '{SUPERVISOR}', which is on every team already.")
        missing = roles - workers
        assert not missing, (
            f"{what} names {sorted(missing)}, but no factory was supplied for "
            f"{'it' if len(missing) == 1 else 'them'}. Worker roles with a factory: "
            f"{sorted(workers)}.")

    required = _role_set(rules.get("required_workers", set()), "required_workers")
    available = _role_set(rules.get("available_workers", workers), "available_workers")
    check_workers(required, "required_workers")
    check_workers(available, "available_workers")
    assert required <= available, (
        f"required_workers {sorted(required - available)} are not in available_workers.")

    restricted_raw = rules.get("restricted_teams", ())
    assert isinstance(restricted_raw, (set, frozenset, list, tuple)), (
        f"restricted_teams must be a collection of teams, got {restricted_raw!r}")
    restricted = set()
    for i, entry in enumerate(restricted_raw):
        team = _role_set(entry, f"restricted_teams[{i}]")
        check_workers(team - {SUPERVISOR}, f"restricted_teams[{i}]")
        restricted.add(team | {SUPERVISOR})

    # Every team the other two rules permit: the Supervisor, the required
    # workers, and any subset of the optional ones.
    optional = sorted(available - required)
    possible = [frozenset({SUPERVISOR, *required,
                          *(w for bit, w in enumerate(optional) if mask >> bit & 1)})
                for mask in range(2 ** len(optional))]

    # A restriction that can never match is almost always a mistake in the rules.
    for team in sorted(restricted - set(possible), key=sorted):
        warn(f"restricted_teams entry {sorted(team)} can never occur under "
             f"required_workers/available_workers, so it restricts nothing.",
             source="topology")

    allowed = tuple(sorted((t for t in possible if t not in restricted),
                           key=lambda t: (len(t), sorted(t))))
    assert allowed, "topology_rules allow no team at all; relax restricted_teams."

    return TopologyRules(required_workers=required, available_workers=available,
                         restricted_teams=frozenset(restricted), allowed_teams=allowed)
