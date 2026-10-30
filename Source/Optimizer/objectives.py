"""The names of NSGA-II's second objective, the one minimized alongside performance.

Kept apart from evolver.py so that runner.py and the reporting code can name the
objective without importing the EA (and with it every agent factory).
"""

# Raw token count across all agents in a team's run. Comparable only when every agent
# uses the same model; once models are mixed a token stops having a fixed price.
OBJECTIVE_TOKEN_COUNT = "token_count"

# USD, each agent priced at its own model's rates from model_config.json.
OBJECTIVE_COST_USD = "cost_usd"

VALID_OBJECTIVES = (OBJECTIVE_TOKEN_COUNT, OBJECTIVE_COST_USD)

# --- display ---------------------------------------------------------------------------
# How the two objectives are named and printed in plots, the console and the CSVs.

# Objective 1, the score NSGA-II maximizes.
PERFORMANCE_LABEL = "ROC-AUC"

# ROC-AUC of a coin flip; drawn as a reference line on every front plot.
PERFORMANCE_NO_SKILL = 0.5

OBJECTIVE_2_LABELS = {
    OBJECTIVE_COST_USD: "Cost (USD)",
    OBJECTIVE_TOKEN_COUNT: "Token count",
}


def format_objective_2(value: float, objective_2: str, decimals: int) -> str:
    """``$0.37`` for cost (``decimals`` places), ``152,340`` for a token count."""
    assert objective_2 in VALID_OBJECTIVES, f"Unknown objective_2: {objective_2}."
    if objective_2 == OBJECTIVE_COST_USD:
        return f"${value:,.{decimals}f}"
    return f"{value:,.0f}"
