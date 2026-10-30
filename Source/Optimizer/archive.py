"""Every unique team evaluated during the run.

The archive ensures that only unique teams are retained, preventing redundant evaluations
and assigning each team a unique identifier that is used to save results and record lineage.

How identity works
------------------
A team's identity is its genome: the roster (which roles are on the team) together with
every role's blueprint (model and full parameter set), plus the run's recursion_limit.
The recursion_limit is not a gene -- it is fixed per run by model_config.json -- but it
bounds how many steps a team may take, so the same roster under a different limit is a
different experiment and must not share scores. The Archive is handed the limit once, at
construction, and folds it into every key. compute_team_key turns the genome into one
deterministic sha256 digest; that digest is the archive's internal dedup key.

What the caller sees is a plain sequential integer id (0, 1, 2, ...), handed out in the
order teams are added.

What each entry stores
----------------------
One entry per unique genome: its id, key, source, roster, and a deepcopy of the evaluated
Team carrying the full multi-evaluation record (performance / token_count / cost /
eval_seconds / metrics / features, one entry per evaluation).

Lineage is not stored here: one genome can be bred several times, from different parents
and variations. Each breeding is a row of the run's generation_members.csv (origin and
parent ids, against these archive ids).

The source says where the record came from: None when this run evaluated the team, or
{"path": ..., "id": ...} when it was replayed from an external archive -- the file and the
team's id in it. Its lineage there is in that run's generation_members.csv under that id.

Reuse (skip re-evaluation)
--------------------------
When the EA produces an offspring whose genome is already archived, the caller checks
contains() and, on a hit, calls apply_results() to replay the stored record onto the fresh
offspring. No LLM calls are made. add() is only ever called for a genuinely new genome.
"""

from typing import Any, Dict, List, Optional, Tuple
from typeguard import typechecked
import copy as cp
import hashlib
import json
import numpy as np
import os

from ..Agents.blueprint import BluePrint
from ..Team_Builder.team import Team
from ..Utils.log import warn


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types.

    Metric and feature payloads can carry numpy values; json.dump cannot serialize
    those, so everything is coerced to native ints/floats/lists before it is written.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def _serialize_blueprint(blueprint: BluePrint) -> Dict[str, Any]:
    """Flatten one role's blueprint into a canonical, JSON-safe dict.

    Used both for the genome key and for the on-disk JSON; the same shape serves both.
    """
    return {
        "model": blueprint.get_model(),
        "parameters": _to_native(blueprint.get_params()),
    }


def _serialize_roster(team: Team) -> Dict[str, Dict[str, Any]]:
    """The team as {role: serialized_blueprint}. The set of keys IS the topology."""
    return {role: _serialize_blueprint(bp) for role, bp in team.get_roster().items()}


def _evaluations(team: Team) -> Dict[str, List[Any]]:
    """The team's per-evaluation record lists, empty for a not-yet-evaluated team.

    Team's record getters assert a non-empty record, so an entry reserve()d
    but not yet finalize()d is read as empty lists here rather than crashing a save.
    """
    if not team.is_evaluated():
        return {k: [] for k in ("performance", "token_count", "cost",
                                "eval_seconds", "metrics", "features")}
    return {
        "performance": team.get_all_performance(),
        "token_count": team.get_all_token_count(),
        "cost": team.get_all_cost(),
        "eval_seconds": team.get_all_eval_seconds(),
        "metrics": team.get_all_metrics(),
        "features": team.get_all_features(),
    }


@typechecked
def compute_team_key(team: Team, recursion_limit: int) -> str:
    """Fold a team's genome and the run's recursion_limit into a sha256 hex digest.

    Serializing with sort_keys makes the digest independent of dict insertion order,
    so two teams built by different code paths hash identically iff they really are
    the same genome under the same step budget.
    """
    canonical = {
        "recursion_limit": recursion_limit,
        "roster": _serialize_roster(team),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@typechecked
class ArchiveRecord:
    """One archived team: its id, source, and full multi-evaluation record."""

    def __init__(self, team_id: int, key: str, team: Team,
                 source: Optional[Dict[str, Any]] = None) -> None:
        self.id = team_id
        self.key = key
        # None: evaluated in this run. Otherwise {"path", "id"} of the external archive
        # entry its record was replayed from.
        self.source = source
        # Deepcopy so later mutation of the caller's object can never rewrite an
        # archived result, and so the record can be replayed onto reused offspring.
        self.team = cp.deepcopy(team)
        return

    def is_evaluated(self) -> bool:
        return self.team.is_evaluated()

    def to_dict(self) -> Dict[str, Any]:
        """A fully JSON-serializable view of this entry, including every evaluation."""
        record = _evaluations(self.team)
        return {
            "id": self.id,
            "key": self.key,
            "source": self.source,
            "roster": _serialize_roster(self.team),
            "n_evaluations": len(record["performance"]),
            **_to_native(record),
        }


@typechecked
class Archive:
    """In-memory store of every unique team genome evaluated during a run.

    Keyed internally by genome digest for O(1) dedup; each entry is handed a
    sequential integer id on the way in.
    """

    def __init__(self, recursion_limit: int) -> None:
        """
        - recursion_limit:
            The run's step budget (ModelConfig.recursion_limit), folded into every
            key and written to the saved JSON so a SeenArchive can tell whether
            another run's scores are comparable.
        """
        assert recursion_limit > 0, f"recursion_limit must be positive: {recursion_limit}."
        self.recursion_limit = recursion_limit
        # genome digest -> ArchiveRecord. One entry per unique genome.
        self._by_key: Dict[str, ArchiveRecord] = {}
        # Next sequential id to hand out. Monotonic; ids are never reused.
        self._next_id: int = 0
        return

    def __len__(self) -> int:
        return len(self._by_key)

    def key(self, team: Team) -> str:
        """The genome digest of ``team`` under this run's recursion_limit."""
        return compute_team_key(team, self.recursion_limit)

    def contains(self, team: Team) -> bool:
        """True iff this genome has already been archived (reserved or evaluated)."""
        return self.key(team) in self._by_key

    def get_id(self, team: Team) -> int:
        """The archived id for an already-present genome (used to record lineage)."""
        key = self.key(team)
        assert key in self._by_key, "Team is not in the archive; cannot resolve its id."
        return self._by_key[key].id

    def get_record(self, team: Team) -> ArchiveRecord:
        """The full entry for an already-present genome."""
        key = self.key(team)
        assert key in self._by_key, "Team is not in the archive; cannot fetch its record."
        return self._by_key[key]

    def _insert(self, key: str, team: Team, source: Optional[Dict[str, Any]] = None) -> int:
        team_id = self._next_id
        self._by_key[key] = ArchiveRecord(team_id=team_id, key=key, team=team, source=source)
        self._next_id += 1
        return team_id

    def add(self, team: Team, source: Optional[Dict[str, Any]] = None) -> int:
        """Archive a newly-evaluated, previously-unseen genome and return its id.

        The one-shot path (evaluate, then archive) used by the serial EA. Asserts the
        genome is absent, so a caller that forgot to check contains() fails loudly
        instead of silently double-counting a team. ``source`` is None for a team this
        run evaluated, or SeenArchive.source_of() for one replayed from another run.
        """
        if source is not None:
            assert set(source) == {"path", "id"}, f"Malformed source: {source}."
        key = self.key(team)
        assert key not in self._by_key, (
            f"Team genome is already in the archive (id={self._by_key[key].id}); "
            f"check contains() before add() and reuse its results instead.")
        assert team.is_evaluated(), (
            "Team has no evaluation results; evaluate it before archiving.")
        return self._insert(key, team, source=source)

    def reserve(self, team: Team) -> int:
        """Assign an id to a genome *before* it is evaluated, and return it.

        The parallel path names each results folder after the id, so the id has to
        exist before the first attempt runs. Reserving a known genome returns the
        existing id (first seen wins). finalize() attaches the record afterwards.
        """
        key = self.key(team)
        if key in self._by_key:
            return self._by_key[key].id
        return self._insert(key, team)

    def finalize(self, team: Team) -> int:
        """Attach an evaluated team's record to its reserved entry and return the id.

        First seen wins: an entry that is already evaluated is left untouched.
        """
        key = self.key(team)
        assert key in self._by_key, "Genome was never reserved; call reserve() before finalize()."
        assert team.is_evaluated(), (
            "Team has no evaluation results; evaluate it before finalizing.")
        record = self._by_key[key]
        if not record.is_evaluated():
            record.team = cp.deepcopy(team)
        return record.id

    def apply_results(self, target: Team) -> int:
        """Replay the archived evaluation record onto a fresh, un-evaluated team.

        Afterwards ``target`` carries an identical multi-evaluation record and flows
        through selection and reporting exactly as a freshly-evaluated team would.
        Returns the archived id of the reused genome.
        """
        key = self.key(target)
        assert key in self._by_key, "Genome is not in the archive; nothing to replay."
        record = self._by_key[key]
        assert record.is_evaluated(), (
            f"Archived genome {record.id} is reserved but not yet evaluated; nothing to replay.")
        assert not target.is_evaluated(), (
            "Target team already has evaluation results; refusing to replay onto a "
            "non-empty record.")

        src = _evaluations(record.team)
        for i in range(len(src["performance"])):
            target.set_evaluation_result(
                src["performance"][i], src["token_count"][i], src["eval_seconds"][i],
                src["metrics"][i], features=src["features"][i], cost=src["cost"][i])
        return record.id

    def records(self) -> List[ArchiveRecord]:
        """All entries, ordered by id (i.e. insertion order)."""
        return sorted(self._by_key.values(), key=lambda r: r.id)

    def to_dict(self) -> Dict[str, Any]:
        """The whole archive as a JSON-serializable dict."""
        return {
            "recursion_limit": self.recursion_limit,
            "n_teams": len(self._by_key),
            "teams": [r.to_dict() for r in self.records()],
        }

    def save_json(self, path: str) -> None:
        """Write the entire archive to a single JSON file at ``path``.

        Written to a temporary file next to ``path`` and then moved over it, so a
        crash mid-write leaves the previous save intact rather than a truncated file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp_path, path)
        return


@typechecked
class SeenArchive:
    """Genomes already evaluated in *other* runs, loaded from their archive JSONs.

    A candidate whose genome matches one of these has its stored record replayed
    instead of costing fresh LLM calls. Read-only history: it hands out no ids and
    records no lineage, since those are per-run.

    Matching uses the same digest as the live Archive, under *this* run's
    recursion_limit, so only teams measured under the same step budget match. When
    one genome appears in several files, the first file wins. Prompts are not part
    of the key: pointing a run at archives produced with different prompt files is
    the caller's responsibility.
    """

    def __init__(self, recursion_limit: int) -> None:
        assert recursion_limit > 0, f"recursion_limit must be positive: {recursion_limit}."
        self.recursion_limit = recursion_limit
        # genome digest -> the per-evaluation record lists needed to replay it.
        self._by_key: Dict[str, Dict[str, Any]] = {}
        # genome digest -> (absolute path of the archive file it was first loaded from,
        # its id in that file), recorded as the source of every team replayed from it.
        self._source: Dict[str, Tuple[str, int]] = {}
        return

    @classmethod
    def from_paths(cls, paths: List[str], recursion_limit: int) -> "SeenArchive":
        """Build a SeenArchive from archive.json paths, in order (earlier files win)."""
        seen = cls(recursion_limit)
        for path in paths:
            seen.load(path)
        return seen

    def load(self, path: str) -> None:
        """Merge one archive.json into the seen set.

        Raises if the path does not exist: a mistyped archive path should fail loudly
        at startup. A file saved under a different recursion_limit (or in the pre-refactor
        format) is loaded but warned about, since none of its keys can match this run.
        Raises on a loaded entry without an integer id: replays record it as their source.
        """
        path = os.path.abspath(path)
        with open(path) as f:
            data = json.load(f)
        if data.get("recursion_limit") != self.recursion_limit:
            warn(f"{path} was saved with recursion_limit={data.get('recursion_limit')} "
                 f"(this run: {self.recursion_limit}); none of its teams will match.",
                 source="archive")
        for entry in data.get("teams", []):
            key = entry.get("key")
            if not key or key in self._by_key or not entry.get("performance"):
                # missing key, first-file-wins on a duplicate genome, or an entry
                # reserved but never evaluated (nothing to replay; evaluate it here)
                continue
            entry_id = entry.get("id")
            assert type(entry_id) is int, (
                f"{path}: team {key} has no integer id ({entry_id!r}); cannot record "
                f"where its replays came from.")
            self._by_key[key] = {k: entry.get(k, []) for k in (
                "performance", "token_count", "cost", "eval_seconds", "metrics", "features")}
            self._source[key] = (path, entry_id)
        return

    def __len__(self) -> int:
        return len(self._by_key)

    def key(self, team: Team) -> str:
        return compute_team_key(team, self.recursion_limit)

    def contains(self, team: Team) -> bool:
        """True iff this genome was evaluated in one of the loaded external archives."""
        return self.key(team) in self._by_key

    def apply_results(self, target: Team) -> None:
        """Replay an external archive's stored record onto a fresh team.

        JSON has only native numbers, so scalars are coerced back to the types
        Team.set_evaluation_result expects. The number of evaluations replayed is
        whatever the source run recorded, which may differ from this run's eval_cnt.
        """
        key = self.key(target)
        assert key in self._by_key, "Genome is not in the seen archive; nothing to replay."
        assert not target.is_evaluated(), (
            "Target team already has evaluation results; refusing to replay onto a "
            "non-empty record.")

        rec = self._by_key[key]
        n = len(rec["performance"])
        assert (len(rec["token_count"]) == len(rec["cost"]) == len(rec["eval_seconds"])
                == len(rec["metrics"]) == len(rec["features"]) == n), (
            f"Malformed seen-archive record for {key}: per-evaluation lists disagree "
            f"in length.")
        for i in range(n):
            target.set_evaluation_result(
                float(rec["performance"][i]),
                int(rec["token_count"][i]),
                float(rec["eval_seconds"][i]),
                dict(rec["metrics"][i] or {}),
                features=list(rec["features"][i] or []),
                cost=float(rec["cost"][i]))
        return

    def source_of(self, team: Team) -> Dict[str, Any]:
        """Where a matched genome's record comes from: the archive file it was first
        loaded from and its id there. Stored as the replayed team's ArchiveRecord.source."""
        key = self.key(team)
        assert key in self._source, "Genome is not in the seen archive; it has no source."
        path, entry_id = self._source[key]
        return {"path": path, "id": entry_id}
