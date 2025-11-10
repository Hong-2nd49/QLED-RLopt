import numpy as np
from .comsol_parser import parse_comsol_csv

class QLEDSimulator:
    """
    Unifies three modes:
      1) Mock analytic model (fast, default)
      2) COMSOL CSV-based evaluation
      3) Surrogate ML model (TODO: via surrogate_model.predict_performance)
    """

    def __init__(self, use_surrogate: bool = False, use_comsol: bool = False):
        self.use_surrogate = use_surrogate
        self.use_comsol = use_comsol

        # Simple metadata for agent; can be refined if you encode states differently.
        self.state_dim = 5
        self.action_dim = 1

    def evaluate(self, design: dict) -> dict:
        # Priority: explicit COMSOL file if provided
        if self.use_comsol and "comsol_csv" in design:
            return parse_comsol_csv(design["comsol_csv"])

        # Placeholder: surrogate model route (to be implemented)
        if self.use_surrogate:
            return self._mock_surrogate(design)

        # Default: mock physics proxy
        return self._mock_physics(design)

    def _mock_physics(self, design: dict) -> dict:
        ZnO_ratio = design["ZnO_ratio"]
        QD_layers = design["QD_layers"]
        ht = design["HTL_thickness_nm"]
        zt = design["ZnO_thickness_nm"]
        bias = design["bias_V"]

        balance_factor = np.exp(-abs(ZnO_ratio - 0.5) * 4.0)
        layer_factor = 0.9 if QD_layers == 2 else 0.8
        thickness_penalty = max(0.0, (bias - 3.5)) * 0.02

        eqe = 0.12 + 0.08 * balance_factor * layer_factor - thickness_penalty
        overlap = 0.6 + 0.3 * balance_factor

        return {
            "EQE": float(max(eqe, 0.0)),
            "recomb_overlap": float(min(overlap, 1.0)),
            "penalty": float(thickness_penalty),
        }

    def _mock_surrogate(self, design: dict) -> dict:
        # For now, just alias mock physics. Replace with real model prediction.
        return self._mock_physics(design)
