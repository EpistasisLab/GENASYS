"""The Azure resource registry: how the R{n}_ credentials are read from the env.

The token-per-minute (TPM) quota on Azure is enforced per deployment within a
resource. The same deployments are replicated across several identical Azure
resources so concurrent evaluations can be spread over them instead of all
competing for one quota.

Each resource ``n`` (1-based) contributes four variables, e.g. for resource 1::

    R1_AZURE_AI_ENDPOINT       # Foundry host for azure_ai + azure_anthropic
    R1_AZURE_AI_CREDENTIAL     # key for the above
    R1_AZURE_OPENAI_ENDPOINT   # Azure OpenAI host
    R1_AZURE_OPENAI_API_KEY    # key for the above

The Azure providers read their credentials only from these prefixed variables;
the unprefixed ``AZURE_*`` names are not used. Resources must be numbered
R1..Rn with no gaps: a gap raises rather than silently dropping every resource
after it.

Every lookup takes an optional ``env`` mapping (``os.environ`` by default) so the
rules can be tested without touching the real environment.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Mapping, Optional, Set, Tuple
from urllib.parse import urlparse

# Variable suffixes, appended to the "R{n}_" prefix.
_AI_ENDPOINT = "AZURE_AI_ENDPOINT"
_AI_CREDENTIAL = "AZURE_AI_CREDENTIAL"
_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
_SUFFIXES = (_AI_ENDPOINT, _AI_CREDENTIAL, _OPENAI_ENDPOINT, _OPENAI_API_KEY)

# provider -> the (endpoint, credential) suffixes it needs from each resource.
# Only the Azure-hosted providers route through a resource; anything else
# (ollama, direct openai/anthropic) keeps its own unprefixed credentials.
_PROVIDER_SUFFIXES = {
    "azure_openai": (_OPENAI_ENDPOINT, _OPENAI_API_KEY),
    "azure_ai": (_AI_ENDPOINT, _AI_CREDENTIAL),
    "azure_anthropic": (_AI_ENDPOINT, _AI_CREDENTIAL),
}
RESOURCE_PROVIDERS = frozenset(_PROVIDER_SUFFIXES)

_RESOURCE_VAR = re.compile(r"^R(\d+)_(" + "|".join(_SUFFIXES) + r")$")


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def _var(n: int, suffix: str) -> str:
    """The env var name for resource ``n`` (1-based) and ``suffix``."""
    assert n >= 1, f"resource ids are 1-based, got {n}"
    return f"R{n}_{suffix}"


def _get(n: int, suffix: str, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Value of ``R{n}_{suffix}``, or None when empty/absent."""
    return _env(env).get(_var(n, suffix)) or None


def detect_resource_count(env: Optional[Mapping[str, str]] = None) -> int:
    """Number of resources R1..Rn defined in the environment (0 when none).

    A resource counts as defined when any of its four variables is non-empty.
    Raises when the numbers are not exactly 1..n, e.g. R1, R2, R4 (no R3) or an
    R0: counting up to the gap would silently drop every resource after it.
    """
    ids = set()
    for name, value in _env(env).items():
        match = _RESOURCE_VAR.match(name)
        if match and value:
            ids.add(int(match.group(1)))
    if not ids:
        return 0
    expected = set(range(1, max(ids) + 1))
    if ids != expected:
        missing = sorted(expected - ids)
        raise ValueError(
            f"Azure resources must be numbered R1..Rn with no gaps; found "
            f"{['R%d' % i for i in sorted(ids)]}"
            + (f", missing {['R%d' % i for i in missing]}" if missing else "")
            + (" (R0 is not allowed, numbering starts at R1)" if 0 in ids else ""))
    return max(ids)


def missing_resource_credentials(providers: Iterable[str],
                                 env: Optional[Mapping[str, str]] = None) -> List[str]:
    """One message per resource variable the run's Azure providers need but lack.

    Every resource R1..Rn must serve every Azure provider in use, since any of
    them may be handed any team. With Azure providers in use and no resources
    defined at all, the message asks for R1.
    """
    provs = sorted({p for p in providers if p in _PROVIDER_SUFFIXES})
    if not provs:
        return []
    n = detect_resource_count(env)
    if n == 0:
        return [f"{', '.join(provs)}: no Azure resources defined; set R1_"
                f"{' / R1_'.join(sorted({s for p in provs for s in _PROVIDER_SUFFIXES[p]}))} "
                f"(and R2_, R3_, ... for more resources)"]
    missing: List[str] = []
    for i in range(1, n + 1):
        for provider in provs:
            for suffix in _PROVIDER_SUFFIXES[provider]:
                if not _get(i, suffix, env):
                    missing.append(f"resource R{i} ({provider}): set {_var(i, suffix)}")
        # Checked once per resource, not per provider: azure_ai and azure_anthropic
        # share the Foundry endpoint.
        endpoint = _get(i, _AI_ENDPOINT, env)
        if endpoint and _AI_ENDPOINT in {s for p in provs for s in _PROVIDER_SUFFIXES[p]}:
            try:
                foundry_root(endpoint, _var(i, _AI_ENDPOINT))
            except ValueError as exc:
                missing.append(f"resource R{i}: {exc}")
    return missing


def foundry_root(endpoint: str, source: str) -> str:
    """``endpoint`` as ``scheme://host``; raises unless it is a bare Foundry resource root.

    The Foundry portal shows per-route URLs (``.../anthropic/v1/messages``), and a
    client given one appends its own route on top and gets a 404. Rather than guess
    which part to drop, any path is refused. ``source`` names where the endpoint came
    from (an env variable or a model entry), for the message.
    """
    parsed = urlparse(endpoint.strip())
    if (not parsed.scheme or not parsed.netloc or parsed.path.strip("/")
            or parsed.query or parsed.fragment):
        raise ValueError(f"{source} must be the Foundry resource root only "
                         f"(https://<resource>.services.ai.azure.com), without a "
                         f"/openai/v1, /anthropic or /models path; got {endpoint!r}. "
                         f"Fix it and rerun.")
    return f"{parsed.scheme}://{parsed.netloc}"


def ai_endpoint(n: int, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``R{n}_AZURE_AI_ENDPOINT``: Foundry host for azure_ai / azure_anthropic."""
    return _get(n, _AI_ENDPOINT, env)


def ai_credential(n: int, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``R{n}_AZURE_AI_CREDENTIAL``: key for azure_ai / azure_anthropic."""
    return _get(n, _AI_CREDENTIAL, env)


def openai_endpoint(n: int, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``R{n}_AZURE_OPENAI_ENDPOINT``: Azure OpenAI host."""
    return _get(n, _OPENAI_ENDPOINT, env)


def openai_api_key(n: int, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``R{n}_AZURE_OPENAI_API_KEY``: key for Azure OpenAI."""
    return _get(n, _OPENAI_API_KEY, env)


class ResourceAllocator:
    """Tracks which ``(resource, model)`` deployments are in use and assigns resources.

    The invariant the parallel evaluation rests on: at most one in-flight evaluation holds
    any ``(resource_id, model)`` pair, so two concurrent evaluations never draw on the same
    deployment's token-per-minute quota. Two evaluations may share a resource when their
    model sets are disjoint, which is what lifts concurrency above the resource count.

    Resource ids are 1-based to match the ``R{n}_`` layout. ``try_assign`` scans low ids
    first, so the grid packs from R1 upward.
    """

    def __init__(self, n_resources: int) -> None:
        assert n_resources >= 1, (
            f"ResourceAllocator needs at least one resource, got {n_resources}; "
            f"callers must skip the allocator when none are configured")
        self.n = int(n_resources)
        self._busy: Set[Tuple[int, str]] = set()

    def try_assign(self, model_set: Iterable[str]) -> Optional[int]:
        """Reserve the lowest resource on which every model in ``model_set`` is free.

        Returns the 1-based resource id, or None when no resource can host the whole set
        right now; the caller leaves the attempt queued until a completion frees something.
        """
        models = {str(m) for m in model_set}
        assert models, "cannot assign a resource to an empty model set"
        for r in range(1, self.n + 1):
            if all((r, m) not in self._busy for m in models):
                self._busy.update((r, m) for m in models)
                return r
        return None

    def release(self, resource_id: int, model_set: Iterable[str]) -> None:
        """Free every ``(resource_id, model)`` pair a finished evaluation held.

        Releasing a pair nobody holds means the reserve/release accounting has drifted, and
        a later assignment would wait on a deployment that is in fact free.
        """
        for m in {str(m) for m in model_set}:
            assert (resource_id, m) in self._busy, (
                f"releasing (R{resource_id}, {m}) that is not marked busy; "
                f"reserve/release accounting has drifted")
            self._busy.discard((resource_id, m))

    def busy_pairs(self) -> int:
        """How many ``(resource, model)`` pairs are reserved right now (0 when idle)."""
        return len(self._busy)
