"""Where a team's evaluation attempts run.

``RayBackend`` runs each attempt as a Ray task on a worker process, which is what a run
uses. ``InlineBackend`` runs each attempt in the calling process, for tests that inspect
the models or stand in for the evaluator. The EA's scheduler needs only the three methods
both classes have: ``submit``, ``wait`` and ``cancel``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

import ray

from ..Team_Builder.team import Team
from .worker import RunContext, evaluate_remote


class RayBackend:
    """Submits attempts to Ray worker processes. The run is put in the object store once,
    rather than once per attempt."""

    def __init__(self, run: RunContext) -> None:
        self._run_ref = ray.put(run)

    def submit(self, team: Team, team_id: int, eval_index: int, resource_id: int) -> Any:
        """Start one attempt; returns the handle ``wait`` and ``cancel`` take."""
        return evaluate_remote.remote(self._run_ref, team, team_id, eval_index, resource_id)

    def wait(self, handles: List[Any], timeout: Optional[float]) -> Optional[Tuple[Any, Any]]:
        """The first finished attempt, as (handle, result). None if none finishes within
        ``timeout`` seconds; with no timeout, waits for one."""
        done, _ = ray.wait(handles, num_returns=1, timeout=timeout)
        if not done:
            return None
        return done[0], ray.get(done[0])

    def cancel(self, handles: List[Any]) -> None:
        """Stop the attempts still running. A queued attempt never starts; a running one is
        interrupted."""
        for handle in handles:
            ray.cancel(handle)


class InlineBackend:
    """Runs each attempt when it is submitted, in this process. ``attempt`` takes
    (team, team_id, eval_index, resource_id) and returns the attempt's result."""

    def __init__(self, attempt: Callable[[Team, int, int, int], Any]) -> None:
        self._attempt = attempt
        self._finished: Deque[Tuple[int, Any]] = deque()
        self._next_handle = 0

    def submit(self, team: Team, team_id: int, eval_index: int, resource_id: int) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self._finished.append((handle, self._attempt(team, team_id, eval_index, resource_id)))
        return handle

    def wait(self, handles: List[Any], timeout: Optional[float]) -> Optional[Tuple[Any, Any]]:
        # Every submitted attempt has already finished, so there is always one to return.
        assert self._finished, "wait() with no attempt in flight"
        return self._finished.popleft()

    def cancel(self, handles: List[Any]) -> None:
        """Nothing to stop: every attempt has already run."""
        return
