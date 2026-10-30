"""Factory for the Statistician, which reviews pending features on the data."""

from __future__ import annotations

from typing import ClassVar, Dict, FrozenSet

from typeguard import typechecked

from .base_factory import WORKER_SYSTEM_PLACEHOLDERS, AgentFactory


@typechecked
class StatisticianFactory(AgentFactory):
    role_name: ClassVar[str] = "statistician"
    display_name: ClassVar[str] = "Statistician"
    call_tool_name: ClassVar[str] = "call_statistician_agent"
    review_fields: ClassVar[Dict[str, str]] = {
        "stats_review": "every review given so far by the Statistician agent"}
    # Review-only: it neither stages, drops nor matches concepts (those flags stay off);
    # it reviews features already mapped, with the statistics tools.
    analyzes_data: ClassVar[bool] = True
    # Review-only: the Supervisor reads this in its roster, so it must not ask the
    # Statistician for new candidates.
    default_description: ClassVar[str] = (
        "reviews features already in pending_features with statistical analysis of "
        "distributions, missingness, prevalence and collinearity; does not suggest "
        "new features")

    required_placeholders: ClassVar[Dict[str, FrozenSet[str]]] = {
        "system": WORKER_SYSTEM_PLACEHOLDERS,
    }
