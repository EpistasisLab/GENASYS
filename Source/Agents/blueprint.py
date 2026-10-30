"""The blueprint of one agent on a team: its evolved genome.

Member variables:
  - model (str): the name of the model to be used (a model_config.json entry)
  - parameters (Dict[str, Any]): the agent's parameters, drawn from that model's param_space

Everything else about the model (provider, deployment, retries, pricing) is fixed per
model in model_config.json and looked up from ModelConfig, so it is not a gene.
"""

from typing import Dict, Any
from typeguard import typechecked
import copy as cp

@typechecked
class BluePrint:
    def __init__(self,
                 model: str,
                 parameters: Dict[str, Any]) -> None:
        """
        Initialize the blueprint with the given parameters.
        - model:
            A model name from model_config.json.
        - parameters:
            The parameter name (key) and value (value) for each knob this agent
            sets, drawn from the model's param_space.
        """

        self.model = cp.deepcopy(model)
        self.parameters = cp.deepcopy(parameters)
        return

    def get_params(self) -> Dict:
        # Return a dictionary containing the parameters of the blueprint.
        # Note that we use deepcopy to ensure that the original parameters are not modified when the dictionary is used.
        return cp.deepcopy(self.parameters)

    # Getters
    def get_model(self) -> str:
        return self.model

    # Setters
    def set_model(self, model: str) -> None:
        self.model = model
