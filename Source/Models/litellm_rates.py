"""Per-model token rates from litellm's maintained price table.

``ModelConfig`` calls ``litellm_rates`` once per model at load time when
``pricing_source`` is ``"litellm"``, and decides what to do with the answer (overlay it,
warn on a disagreement with the config). This module only looks rates up.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import litellm

# provider -> litellm price-table key prefix. A provider absent here is never priced
# from litellm (e.g. ollama is local and genuinely free).
LITELLM_PREFIXES: Dict[str, str] = {
    "azure_openai": "azure/",
    "azure_ai": "azure_ai/",
    # Foundry-hosted Claude bills as Foundry, not as Anthropic direct.
    "azure_anthropic": "azure_ai/",
    "anthropic": "",
    "openai": "",
}


@lru_cache(maxsize=1)
def _litellm_table() -> Dict[str, Any]:
    """litellm's price map, copied once per process."""
    return dict(litellm.model_cost)


def litellm_rates(provider: str, name: str, azure_deployment: Optional[str]
                  ) -> Optional[Tuple[float, float, Optional[float]]]:
    """(input, output, cached_input) USD per million from litellm, or None.

    litellm stores dollars per token, so the per-million conversion happens here.
    """
    if provider not in LITELLM_PREFIXES:
        return None
    prefix = LITELLM_PREFIXES[provider]
    table = _litellm_table()
    # Try the model name first; fall back to the deployment name for the common case
    # where the deployment was named after its model.
    for key in (name, azure_deployment):
        if not key:
            continue
        entry = table.get(f"{prefix}{key}") or table.get(str(key))
        if entry and entry.get("input_cost_per_token") is not None:
            cached = entry.get("cache_read_input_token_cost")
            return (float(entry["input_cost_per_token"]) * 1e6,
                    float(entry.get("output_cost_per_token") or 0.0) * 1e6,
                    None if cached is None else float(cached) * 1e6)
    return None
