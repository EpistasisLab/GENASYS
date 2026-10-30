"""The ``print_dataset_schema`` tool: what the raw table holds.

The catalog says what a feature means; this says what is in the table: every column
with its dtype and an example value, plus three sample rows. It reads
``ToolResources.dataset``, the dev split, so the holdout never reaches an agent, and
leaves out the target, which an agent would otherwise propose as a feature. Offered as a
tool rather than rendered into a prompt, so a run that never needs the schema never
spends context on it.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from typeguard import typechecked

from ...Task.task import render_schema_description
from .tool_resources import ToolResources

PRINT_DATASET_SCHEMA = "print_dataset_schema"


@typechecked
def dataset_schema_tool(resources: ToolResources) -> BaseTool:
    # Rendered once: the dev split is fixed for the run.
    schema = render_schema_description(resources.dataset, resources.target_column)

    @tool(PRINT_DATASET_SCHEMA)
    def print_dataset_schema() -> str:
        """Describe the dataset: every available column with its dtype and an example value, plus three sample rows. The prediction target is excluded. Use when you need to know what the data actually contains before proposing features."""
        return schema

    return print_dataset_schema
