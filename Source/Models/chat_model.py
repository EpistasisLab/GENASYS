"""Build one agent's chat model from its blueprint and the run's ModelConfig.

The blueprint names the model and carries its evolved parameters (drawn from the
model's ``param_space``); everything else about the model (provider, Azure deployment,
API version, endpoint, timeout) is fixed per model in model_config.json. The
parameters are forwarded as-is: ``param_space`` only offers knobs the model's API
accepts.

Credentials:
    Read only from the environment (.env is loaded by ModelConfig). The direct
    ``openai`` / ``anthropic`` providers read their unprefixed key themselves, so it
    never passes through this module. The Azure providers read from resource
    ``R{n}_`` (see ``resources.py``); their key is wrapped (``SecretStr`` /
    ``AzureKeyCredential``) before it enters any argument dict, so no repr,
    traceback or log of the arguments or the built model shows it. Error messages
    name variables, never values.

Each Azure call goes to one resource, chosen by the caller (``resource``). There is
one resource per run until evaluations run in parallel.

``check_param_names`` builds every configured model offline, with stand-in
credentials, and raises on a ``param_space`` knob its client does not recognize;
the EA runs it at startup. Whether the API accepts a recognized knob's values is
only known by calling it (``EA.check_models``).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from azure.core.credentials import AzureKeyCredential
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr
from typeguard import typechecked

from ..Agents.blueprint import BluePrint
from . import resources
from .credentials import package_for_provider
from .model_config import CONNECTION_ARGUMENTS, ModelConfig, ModelSpec

# Providers that are a hosting arrangement rather than a wire protocol: Claude on
# Azure AI Foundry speaks Anthropic's Messages API, so it is built as ``anthropic``
# with an Azure base URL and key.
_INIT_PROVIDER = {"azure_anthropic": "anthropic"}

# Stand-in credentials for check_param_names: a client needs a key and an endpoint to
# be built, but nothing is sent. The real environment is never read for that check.
_PLACEHOLDER_KEY = "placeholder"
_PLACEHOLDER_ENV = {"R1_AZURE_OPENAI_ENDPOINT": "https://placeholder.openai.azure.com",
                    "R1_AZURE_OPENAI_API_KEY": _PLACEHOLDER_KEY,
                    "R1_AZURE_AI_ENDPOINT": "https://placeholder.services.ai.azure.com",
                    "R1_AZURE_AI_CREDENTIAL": _PLACEHOLDER_KEY}


@typechecked
def build_chat_model(blueprint: BluePrint, model_config: ModelConfig, resource: int,
                     env: Optional[Mapping[str, str]] = None) -> BaseChatModel:
    """The chat model ``blueprint`` describes, connected through Azure resource ``resource``.

    Args:
        resource: the 1-based ``R{n}_`` resource the Azure providers connect through;
            unused by the others.
        env: where the credentials are read from; ``os.environ`` by default.
    """
    spec, kwargs = _arguments(blueprint, model_config, resource, env)
    return _init(spec, kwargs)


@typechecked
def knob_settings(param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Settings that between them use every value in ``param_space``: every knob at its
    first value, then that with one knob changed to each of its other values. A team
    sends all of a model's knobs at once, so every setting also tests the knobs
    together. A model with no knobs has one setting, ``{}``."""
    base = {knob: values[0] for knob, values in param_space.items()}
    settings = [base]
    for knob, values in param_space.items():
        settings.extend({**base, knob: value} for value in values[1:])
    return settings


@typechecked
def check_param_names(model_config: ModelConfig) -> None:
    """Build every model in ``model_config`` with each of its ``knob_settings``, offline,
    and raise listing every knob its client does not recognize, and every setting the
    client refuses to be built with.

    An unrecognized knob would otherwise fail at a team's first call (OpenAI and
    Anthropic forward it to the API as an unknown argument) or be silently dropped
    (Ollama). A knob counts as recognized when it is a field of the client class.
    Nothing is sent and the real credentials are not read: the Azure providers get
    ``_PLACEHOLDER_ENV``, the direct ones a placeholder key.
    """
    problems: List[str] = []
    for model in model_config.model_pool():
        for params in knob_settings(model_config.param_space(model)):
            spec, kwargs = _arguments(BluePrint(model=model, parameters=params),
                                      model_config, 1, _PLACEHOLDER_ENV)
            if spec.provider in ("openai", "anthropic"):
                kwargs["api_key"] = SecretStr(_PLACEHOLDER_KEY)
            try:
                # The clients warn about an unknown knob; it is reported below instead.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chat = _init(spec, kwargs)
            except Exception as exc:  # noqa: BLE001 - every refused setting is reported
                problems.append(f"'{model}' with {params}: {type(exc).__name__}: {exc}")
                continue
            fields = type(chat).model_fields
            known = set(fields) | {f.alias for f in fields.values() if f.alias}
            unknown = sorted(set(params) - known)
            if unknown:
                problems.append(f"'{model}': {unknown} not recognized by "
                                f"{type(chat).__name__} (provider '{spec.provider}')")
    if problems:
        raise ValueError(
            f"{model_config.source}: param_space settings the chat clients do not accept "
            f"(fix the knob name, or remove it):\n  " + "\n  ".join(dict.fromkeys(problems)))


def _arguments(blueprint: BluePrint, model_config: ModelConfig, resource: int,
               env: Optional[Mapping[str, str]]) -> Tuple[ModelSpec, Dict[str, Any]]:
    """The model's spec and the ``init_chat_model`` arguments for ``blueprint``."""
    spec = model_config.get(blueprint.get_model())
    connect = _CONNECTIONS.get(spec.provider)
    if connect is None:
        raise ValueError(f"model '{spec.name}': provider '{spec.provider}' is not supported; "
                         f"supported providers are {sorted(_CONNECTIONS)}")

    params = blueprint.get_params()
    # ModelConfig refuses these in param_space; a blueprint built by hand could still
    # carry one, and it would replace part of the connection.
    copies = sorted(CONNECTION_ARGUMENTS & set(params))
    assert not copies, f"model '{spec.name}': parameters {copies} copy connection arguments"

    kwargs: Dict[str, Any] = {"model": spec.name,
                              "model_provider": _INIT_PROVIDER.get(spec.provider,
                                                                   spec.provider)}
    # Drawn with numpy; the clients want native Python values.
    for name, value in params.items():
        kwargs[name] = value.item() if isinstance(value, np.generic) else value
    connect(spec, kwargs, model_config, resource, env)
    return spec, kwargs


def _init(spec: ModelSpec, kwargs: Dict[str, Any]) -> BaseChatModel:
    try:
        return init_chat_model(**kwargs)
    except ImportError as exc:
        package = package_for_provider(spec.provider)
        raise ImportError(f"provider '{spec.provider}' (model '{spec.name}') is not "
                          f"installed; pip install {package}") from exc


# ------------------------------------ connections -------------------------------------------

def _azure_openai(spec: ModelSpec, kwargs: Dict[str, Any], config: ModelConfig,
                  resource: int, env: Optional[Mapping[str, str]]) -> None:
    """Azure OpenAI: routed by deployment name, which need not equal the model name.
    ModelConfig has checked the deployment and ``api_version`` are set."""
    kwargs.update(azure_deployment=spec.azure_deployment, api_version=spec.api_version,
                  azure_endpoint=spec.endpoint or _require(
                      resources.openai_endpoint(resource, env), resource,
                      "AZURE_OPENAI_ENDPOINT"),
                  api_key=SecretStr(_require(resources.openai_api_key(resource, env),
                                             resource, "AZURE_OPENAI_API_KEY")),
                  timeout=spec.request_timeout_s, max_retries=config.default_max_retries)


def _azure_ai(spec: ModelSpec, kwargs: Dict[str, Any], config: ModelConfig,
              resource: int, env: Optional[Mapping[str, str]]) -> None:
    """Azure AI Foundry's OpenAI-compatible route (GPT, Llama, DeepSeek, Mistral; not Claude).

    The client has no deployment field: on this route the deployment name is sent as
    the model. The endpoint is the resource root plus ``/openai/v1``.
    """
    endpoint = spec.endpoint or _require(resources.ai_endpoint(resource, env), resource,
                                         "AZURE_AI_ENDPOINT")
    kwargs.update(model=spec.azure_deployment,
                  endpoint=_foundry_root(spec, endpoint, resource) + "/openai/v1",
                  credential=AzureKeyCredential(_require(
                      resources.ai_credential(resource, env), resource,
                      "AZURE_AI_CREDENTIAL")),
                  timeout=spec.request_timeout_s, max_retries=config.default_max_retries)
    if spec.api_version:
        kwargs["api_version"] = spec.api_version


def _azure_anthropic(spec: ModelSpec, kwargs: Dict[str, Any], config: ModelConfig,
                     resource: int, env: Optional[Mapping[str, str]]) -> None:
    """Claude on Azure AI Foundry, through Anthropic's own Messages API.

    Foundry serves Claude only on ``/anthropic`` (its ``/openai/v1`` route answers 404),
    so this is ``ChatAnthropic`` pointed at the Foundry host, with the deployment name
    sent as the model. The SDK appends ``/v1/messages`` itself.
    """
    endpoint = spec.endpoint or _require(resources.ai_endpoint(resource, env), resource,
                                         "AZURE_AI_ENDPOINT")
    kwargs.update(model=spec.azure_deployment,
                  base_url=_foundry_root(spec, endpoint, resource) + "/anthropic",
                  api_key=SecretStr(_require(resources.ai_credential(resource, env),
                                             resource, "AZURE_AI_CREDENTIAL")),
                  timeout=spec.request_timeout_s, max_retries=config.default_max_retries)


def _direct(spec: ModelSpec, kwargs: Dict[str, Any], config: ModelConfig,
            resource: int, env: Optional[Mapping[str, str]]) -> None:
    """OpenAI's or Anthropic's own API: the client reads OPENAI_API_KEY /
    ANTHROPIC_API_KEY from the environment itself."""
    kwargs.update(timeout=spec.request_timeout_s, max_retries=config.default_max_retries)


def _ollama(spec: ModelSpec, kwargs: Dict[str, Any], config: ModelConfig,
            resource: int, env: Optional[Mapping[str, str]]) -> None:
    """A local or remote Ollama server: ``endpoint`` in model_config.json, or the
    client's own default (localhost). ChatOllama has no retries; the timeout goes to
    its HTTP client. ``num_ctx`` is a param_space knob like any other."""
    if spec.endpoint:
        kwargs["base_url"] = spec.endpoint
    kwargs["client_kwargs"] = {"timeout": spec.request_timeout_s}


_CONNECTIONS: Dict[str, Callable[..., None]] = {
    "azure_openai": _azure_openai,
    "azure_ai": _azure_ai,
    "azure_anthropic": _azure_anthropic,
    "openai": _direct,
    "anthropic": _direct,
    "ollama": _ollama,
}


# -------------------------------------- helpers ---------------------------------------------

def _require(value: Optional[str], resource: int, suffix: str) -> str:
    """``value`` of ``R{resource}_{suffix}``; raises naming the variable when unset.

    The credential check at run start should have caught this already.
    """
    if not value:
        raise RuntimeError(f"R{resource}_{suffix} is not set; put it in the environment "
                           f"or in .env (see .env.example)")
    return value


def _foundry_root(spec: ModelSpec, endpoint: str, resource: int) -> str:
    """``endpoint`` as ``scheme://host``. Both sources were checked at run start
    (ModelConfig, ``require_credentials``); checked again here for direct callers."""
    return resources.foundry_root(endpoint, f"model '{spec.name}': endpoint"
                                  if spec.endpoint else f"R{resource}_AZURE_AI_ENDPOINT")
