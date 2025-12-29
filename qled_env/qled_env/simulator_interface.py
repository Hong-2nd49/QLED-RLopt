from __future__ import annotations
from typing import Dict, Any


class SimulatorInterface:
    """
    Minimal simulator protocol for QLED-RL.

    Implementations may be:
    - surrogate model predictor (fast)
    - COMSOL batch runner (slow, accurate)
    """

    def evaluate(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Args:
            params: real-parameter dict (physical units)

        Returns:
            metrics dict, e.g.:
              {
                "eqe": float,
                "luminance": float,
                "power_eff": float,
                "uncertainty": float (optional),
              }
        """
        raise NotImplementedError("SimulatorInterface.evaluate() must be implemented")
