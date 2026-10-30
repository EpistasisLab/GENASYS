"""Provider credential checks and package lookups.

Used by ModelConfig and run setup before the expensive work starts.
``require_credentials`` fails fast when a named provider lacks its env vars,
``package_for_provider`` names the pip package for an error, and
``load_dotenv_once`` loads a .env into os.environ before either check.

The Azure providers take their credentials from the numbered R{n}_ resources
(see ``resources``); every other provider from the unprefixed names below.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from dotenv import load_dotenv

from .resources import RESOURCE_PROVIDERS, missing_resource_credentials

# Env vars each non-Azure provider needs. Each value is a tuple of groups; a
# group is satisfied when any one name in it is set. Only checked for providers a
# run uses. The Azure providers are checked per resource in ``resources`` instead.
_PROVIDER_CREDENTIALS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "anthropic": (("ANTHROPIC_API_KEY",),),
    "openai": (("OPENAI_API_KEY",),),
    "ollama": (),
}

# pip package serving each provider, named in the missing-install error.
_PROVIDER_PACKAGES: Dict[str, str] = {
    "azure_openai": "langchain-openai",
    "openai": "langchain-openai",
    "azure_ai": "langchain-azure-ai",
    "azure_anthropic": "langchain-anthropic",
    "anthropic": "langchain-anthropic",
    "bedrock_converse": "langchain-aws",
    "google_anthropic_vertex": "langchain-google-vertexai",
    "ollama": "langchain-ollama",
}


@lru_cache(maxsize=1)
def load_dotenv_once() -> None:
    """Load a .env into os.environ once per process.

    ``override=False`` keeps an already-exported variable (SLURM, CI, the shell)
    ahead of the file.
    """
    load_dotenv(override=False)


def require_credentials(providers: Iterable[str],
                        env: Optional[Mapping[str, str]] = None) -> None:
    """Raise if any named provider has no credentials in the env.

    Called before the expensive setup (data load, Ray init) so a missing key costs
    a second, not a generation. Only providers the run names are checked: the
    Azure ones on every resource R1..Rn, the rest by their unprefixed names.
    ``env`` defaults to ``os.environ`` (after loading .env).
    """
    if env is None:
        load_dotenv_once()
        env = os.environ
    provs = sorted(set(p for p in providers if p))
    missing: List[str] = missing_resource_credentials(provs, env)
    for provider in provs:
        if provider in RESOURCE_PROVIDERS:
            continue
        for group in _PROVIDER_CREDENTIALS.get(provider, ()):
            if not any(env.get(name) for name in group):
                missing.append(f"{provider}: set one of {' / '.join(group)}")
    if missing:
        raise RuntimeError(
            "Missing model provider credentials:\n  " + "\n  ".join(missing)
            + "\nPut them in the environment or in a .env file (see .env.example).")


def package_for_provider(provider: str) -> str:
    """The pip package that serves ``provider``, or "" if unknown."""
    return _PROVIDER_PACKAGES.get(provider, "")
