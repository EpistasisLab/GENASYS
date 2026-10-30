"""Per-model provider, pricing, and parameter-space configuration from one JSON file.

``ModelSpec`` holds one model's routing, pricing, and param_space; ``ModelConfig``
parses the file and answers lookups for the EA (evolver, reproduction) and for cost
accounting. The config path is required and must exist, so a missing file fails
loudly instead of degrading to zero pricing. Companion modules: ``litellm_rates``
(litellm's price table), ``credentials`` (provider keys, .env), ``resources`` (Azure
resource variables).

Dollars are the second NSGA-II objective because raw token counts are not comparable
across models that differ in price; ``price_agent`` turns one agent's tokens into USD.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set, Tuple

from ..Utils.log import warn
from .credentials import load_dotenv_once
from .litellm_rates import LITELLM_PREFIXES, litellm_rates
from .resources import foundry_root

# Where per-token rates come from: litellm's maintained price table, or only the
# numbers written in the config file.
PRICING_SOURCE_LITELLM = "litellm"
PRICING_SOURCE_CONFIG = "config"
VALID_PRICING_SOURCES = (PRICING_SOURCE_LITELLM, PRICING_SOURCE_CONFIG)

# Chat-model arguments set from a model's entry and the environment (see
# chat_model.py). A param_space knob of the same name would be a second copy
# of the setting, and one of the two would silently win.
CONNECTION_ARGUMENTS = frozenset({
    "model", "model_provider", "max_retries", "timeout", "azure_deployment",
    "api_version", "azure_endpoint", "api_key", "endpoint", "credential", "base_url",
    "client_kwargs"})

# Providers reached through an Azure AI Foundry resource root.
_FOUNDRY_PROVIDERS = ("azure_ai", "azure_anthropic")


def _whole_number(what: str, value: Any, minimum: int) -> int:
    """``value`` if it is an int >= ``minimum``. A float or a bool is rejected
    rather than truncated: int(2.5) and int(True) would silently run a different
    setting than the one written."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be a whole number, got {value!r}")
    if value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}")
    return value


def _positive_seconds(what: str, value: Any) -> float:
    """``value`` as a float if it is a number > 0; a zero or negative timeout
    would fail every request."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number of seconds, got {value!r}")
    if value <= 0:
        raise ValueError(f"{what} must be > 0, got {value}")
    return float(value)


def _non_negative_seconds(what: str, value: Any) -> float:
    """``value`` as a float if it is a number >= 0; 0 means no wait."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number of seconds, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must be >= 0, got {value}")
    return float(value)


def _rates_agree(a: float, b: float) -> bool:
    """True when two per-million rates match to a hundredth of a cent."""
    return abs(a - b) < 1e-4


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """A JSON object as a dict; raises on a repeated key, which json.load would
    otherwise resolve by silently keeping the last copy."""
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"model config: key '{key}' appears more than once in the "
                             f"same object; keep one copy")
        out[key] = value
    return out


@dataclass(frozen=True)
class ModelSpec:
    """Everything ``build_chat_model`` and the cost accounting need about one model."""

    name: str
    provider: str
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    endpoint: Optional[str] = None

    # USD per million tokens. Zero = free (a lie for a paid model, hence ``priced``).
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    # Discounted cached-input rate. None => bill cached input at the full input rate.
    cached_input_per_million: Optional[float] = None

    # knob name -> allowed values the EA may set/mutate. A knob absent here is never
    # set for this model. Seeds the initial population and bounds mutations.
    param_space: Dict[str, List[Any]] = field(default_factory=dict)

    request_timeout_s: Optional[float] = None

    @property
    def priced(self) -> bool:
        return bool(self.input_per_million or self.output_per_million)

    @classmethod
    def from_entry(cls, name: str, entry: Dict[str, Any],
                   default_timeout: Optional[float]) -> "ModelSpec":
        """Build a spec from one ``models`` entry of the config."""
        rates = (float(entry.get("input_per_million", 0.0) or 0.0),
                 float(entry.get("output_per_million", 0.0) or 0.0))
        for label, rate in zip(("input_per_million", "output_per_million"), rates):
            if rate < 0:
                raise ValueError(f"model '{name}': {label} must not be negative (got {rate})")

        cached = entry.get("cached_input_per_million")
        if cached is not None and float(cached) < 0:
            raise ValueError(f"model '{name}': cached_input_per_million must not be negative")

        return cls(
            name=name,
            provider=str(entry.get("provider", "") or ""),
            azure_deployment=entry.get("azure_deployment") or None,
            api_version=entry.get("api_version") or None,
            endpoint=entry.get("endpoint") or None,
            input_per_million=rates[0],
            output_per_million=rates[1],
            cached_input_per_million=None if cached is None else float(cached),
            param_space=cls._validate_param_space(name, entry.get("param_space")),
            # Optional per-model override of the run-wide request_timeout_s.
            request_timeout_s=(_positive_seconds(f"model '{name}': request_timeout_s",
                                                 entry["request_timeout_s"])
                               if entry.get("request_timeout_s") is not None else default_timeout),
        )

    @staticmethod
    def _validate_param_space(name: str, param_space: Any) -> Dict[str, List[Any]]:
        """Check a model's param_space: each knob maps to a non-empty list.

        The key is required: a model with nothing to evolve must say so with an
        explicit ``{}``, so a misspelled key is not silently read as "no knobs".
        Validated here so the error names the model, not a stack inside population
        seeding.
        """
        if param_space is None:
            raise ValueError(f"model '{name}': missing 'param_space' (use {{}} if the "
                             f"model has no parameters to evolve)")
        if not isinstance(param_space, dict):
            raise ValueError(f"model '{name}': param_space must be an object of "
                             f"knob -> list-of-values")
        copies = sorted(CONNECTION_ARGUMENTS & set(param_space))
        if copies:
            raise ValueError(
                f"model '{name}': param_space knob(s) {copies} copy a connection "
                f"argument that is set from the model's entry and the environment; "
                f"remove them from param_space")
        out: Dict[str, List[Any]] = {}
        for knob, values in param_space.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"model '{name}': param_space['{knob}'] must be a non-empty list "
                    f"(got {values!r})")
            out[str(knob)] = list(values)
        return out


class ModelConfig:
    """Parsed model_config.json plus the CLI back-fills, held by the EA for a run.

    The config path is the only required constructor argument; the class loads any
    ``.env`` and parses the file itself. The path must exist. ``azure_deployment`` /
    ``api_version`` fill a hole in a model's entry but never override a value it
    already sets. No process-wide singleton — the caller constructs one and holds it.
    """

    # Placeholder deployment name in the shipped config; an unedited one raises at
    # load (checked in _resolve_spec) unless a CLI deployment fills it.
    _PLACEHOLDER: ClassVar[str] = "REPLACE-ME"

    # Providers that route by an Azure deployment name.
    _AZURE_PROVIDERS: ClassVar[Tuple[str, ...]] = (
        "azure_openai", "azure_ai", "azure_anthropic")

    # Top-level keys model_config.json must set; none has a default.
    _REQUIRED_SETTINGS: ClassVar[Tuple[str, ...]] = (
        "pricing_source", "default_max_retries", "request_timeout_s",
        "recursion_limit", "team_retry_attempts", "team_retry_delay_secs",
        "abort_after_consecutive_infra_failures", "models")

    def __init__(self,
                 config_path: str | Path,
                 *,
                 azure_deployment: Optional[str] = None,
                 api_version: Optional[str] = None,
                 verbose: bool = True) -> None:

        load_dotenv_once()
        path, raw = self._read(config_path)
        self.source = str(path)

        # Every run-wide setting is required: a silent default would change how a
        # run behaves (and, for recursion_limit, which archives it matches)
        # without the config saying so.
        missing = [key for key in self._REQUIRED_SETTINGS if key not in raw]
        if missing:
            raise ValueError(f"{path}: missing required setting(s) {missing}")

        self.pricing_source = str(raw["pricing_source"])
        if self.pricing_source not in VALID_PRICING_SOURCES:
            raise ValueError(
                f"pricing_source must be one of {VALID_PRICING_SOURCES}, "
                f"got '{self.pricing_source}'")

        # Provider-SDK retries per request.
        self.default_max_retries = _whole_number(
            "default_max_retries", raw["default_max_retries"], minimum=1)
        self.request_timeout_s = _positive_seconds(
            "request_timeout_s", raw["request_timeout_s"])
        # Graph super-steps a team may take; also part of every archive key.
        self.recursion_limit = _whole_number(
            "recursion_limit", raw["recursion_limit"], minimum=1)
        # Re-runs of a whole team after a provider failure, on top of the first
        # attempt; 0 means never retry.
        self.team_retry_attempts = _whole_number(
            "team_retry_attempts", raw["team_retry_attempts"], minimum=0)
        # Wait before each of those re-runs, so a rate limit has time to clear.
        self.team_retry_delay_secs = _non_negative_seconds(
            "team_retry_delay_secs", raw["team_retry_delay_secs"])
        # Consecutive provider-failed teams before the run aborts.
        self.abort_after_consecutive_infra_failures = _whole_number(
            "abort_after_consecutive_infra_failures",
            raw["abort_after_consecutive_infra_failures"], minimum=1)
        # Wall-clock cap on one team run, which then counts as truncated. Optional, unlike
        # the settings above: leaving it out means no cap, which is how runs behaved
        # before it existed.
        self.team_time_limit_s: Optional[float] = (
            _positive_seconds("team_time_limit_s", raw["team_time_limit_s"])
            if raw.get("team_time_limit_s") is not None else None)

        if not isinstance(raw["models"], dict) or not raw["models"]:
            raise ValueError(f"{path}: 'models' must be a non-empty object of "
                             f"model name -> entry")

        # CLI back-fills: fill a hole in a model's entry, never override a set value.
        # Set before the models loop because _resolve_spec reads them.
        self.cli_azure_deployment: Optional[str] = azure_deployment or None
        self.cli_api_version: Optional[str] = api_version or None

        # Each spec is fully resolved here (CLI back-fills, Azure defaults, rates) so
        # get() is a pure lookup: pricing runs per agent in Ray workers, and the
        # back-fills are deterministic per spec, so neither belongs on that hot path.
        self.models: Dict[str, ModelSpec] = {}
        for name, entry in raw["models"].items():
            if not isinstance(entry, dict):
                raise ValueError(f"model '{name}': entry must be an object")
            spec = ModelSpec.from_entry(str(name), entry,
                                        default_timeout=self.request_timeout_s)
            # azure_openai needs a deployment name (not derivable from the model
            # name); catch a missing one at load, not at request time.
            if spec.provider == "azure_openai" and not spec.azure_deployment:
                raise ValueError(
                    f"model '{name}': provider 'azure_openai' requires 'azure_deployment' "
                    f"(the Azure deployment name, which is not the same as the model name)")
            if spec.provider == "azure_openai" and not spec.api_version:
                raise ValueError(
                    f"model '{name}': provider 'azure_openai' requires 'api_version' "
                    f"(e.g. \"2024-10-21\"); set it in the model's entry")
            if spec.provider in _FOUNDRY_PROVIDERS and spec.endpoint:
                foundry_root(spec.endpoint, f"model '{name}': endpoint")
            # Resolved before pricing so a CLI-filled deployment name can match
            # litellm's table.
            spec = self._resolve_spec(str(name), spec)
            if self.pricing_source == PRICING_SOURCE_LITELLM:
                spec = self._price_from_litellm(str(name), spec)
            self.models[str(name)] = spec

        if verbose:
            print(f"    [model_config] loaded {path} "
                  f"({len(self.models)} models, model pool {self.model_pool()})",
                  flush=True)

    @staticmethod
    def _read(config_path: str | Path) -> Tuple[Path, Dict[str, Any]]:
        """Load the JSON at ``config_path``.

        Required and must exist: a missing file fails here rather than degrading to
        zero pricing, which would make every paid model look free to the EA.
        """
        if not config_path:
            raise ValueError("ModelConfig requires a path to a model_config.json")
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"model config not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            return path, json.load(handle, object_pairs_hook=_no_duplicate_keys)

    # --- lookup -------------------------------------------------------------

    def _resolve_spec(self, model: str, spec: ModelSpec) -> ModelSpec:
        """Apply CLI back-fills and Azure deployment defaults; called once at load.

        Resolving here (not in get()) means each spec is finalized when the config is
        built. An unedited placeholder deployment raises now rather than at first use.
        """
        # CLI defaults fill holes only; a real entry wins. An unedited placeholder
        # counts as a hole so `--azure_deployment` can fill it.
        placeholder = spec.azure_deployment == self._PLACEHOLDER
        if (not spec.azure_deployment or placeholder) and self.cli_azure_deployment:
            spec = replace(spec, azure_deployment=self.cli_azure_deployment)
            placeholder = False
        if not spec.api_version and self.cli_api_version:
            spec = replace(spec, api_version=self.cli_api_version)

        if placeholder:
            raise RuntimeError(
                f"model '{model}' still has azure_deployment '{self._PLACEHOLDER}' in "
                f"{self.source}. Replace it with your Azure deployment name, or pass "
                f"--azure_deployment on the command line.")

        # Azure needs some deployment name; the model name works when the deployment
        # was named after its model.
        if spec.provider in self._AZURE_PROVIDERS and not spec.azure_deployment:
            spec = replace(spec, azure_deployment=spec.name)
        return spec

    def get(self, model: str) -> ModelSpec:
        """The fully-resolved spec for ``model``, which must have a config entry.

        model_config.json defines the full set of evolvable models, so a model with
        no entry is a misconfiguration and raises rather than being guessed.
        """
        spec = self.models.get(model)
        if spec is None:
            raise KeyError(
                f"model '{model}' has no entry in {self.source}; configured models "
                f"are {self.model_pool()}. Add it to model_config.json before use.")
        return spec

    def provider_of(self, model: str) -> str:
        return self.get(model).provider

    def model_pool(self) -> List[str]:
        """Models the EA may evolve toward, sorted for reproducible seeded runs."""
        return sorted(self.models)

    def providers_in_use(self, models: Iterable[str]) -> Set[str]:
        return {self.provider_of(m) for m in models if m}

    def param_space(self, model: str) -> Dict[str, List[Any]]:
        """The discrete parameter space the EA may draw from for ``model``."""
        return dict(self.get(model).param_space)

    # --- pricing ------------------------------------------------------------

    def price_agent(self, model: str, input_tokens: int, output_tokens: int,
                    cached_input_tokens: int = 0) -> float:
        """USD for one agent's token usage under ``model``'s rates."""
        spec = self.get(model)
        cached = max(0, int(cached_input_tokens))
        billable_input = max(0, int(input_tokens) - cached)
        cached_rate = (spec.input_per_million if spec.cached_input_per_million is None
                       else spec.cached_input_per_million)
        return (billable_input * spec.input_per_million / 1_000_000.0
                + cached * cached_rate / 1_000_000.0
                + max(0, int(output_tokens)) * spec.output_per_million / 1_000_000.0)

    @staticmethod
    def _price_from_litellm(name: str, spec: ModelSpec) -> ModelSpec:
        """Overlay litellm's rates onto ``spec``, keeping the config's numbers as fallback.

        A config/litellm disagreement is warned because with objective_2 = cost_usd a
        stale rate silently biases the Pareto front; so is a model neither prices. Each
        model is priced once, at load, so each warning shows once per run.
        """
        rates = litellm_rates(spec.provider, spec.name, spec.azure_deployment)
        if rates is None:
            if spec.provider in LITELLM_PREFIXES and not spec.priced:
                warn(f"litellm has no price for '{name}' (provider '{spec.provider}') "
                     f"and the config gives none either — it will be costed at ZERO.")
            return spec

        in_m, out_m, cached_m = rates
        if spec.priced and not (_rates_agree(spec.input_per_million, in_m)
                                and _rates_agree(spec.output_per_million, out_m)):
            warn(f"'{name}': config says ${spec.input_per_million:.4f}/"
                 f"${spec.output_per_million:.4f} per million but litellm says "
                 f"${in_m:.4f}/${out_m:.4f}. Using litellm's. Update the config so "
                 f"its fallback rates are right too.")
        return replace(spec, input_per_million=in_m, output_per_million=out_m,
                       cached_input_per_million=cached_m)

    def has_priced_models(self) -> bool:
        """True if any model-pool model carries a non-zero rate.

        The caller uses this to guard a cost_usd objective: with nothing priced,
        cost is constant 0.0 and NSGA-II collapses to single-objective.
        """
        return any(spec.priced for spec in self.models.values())
