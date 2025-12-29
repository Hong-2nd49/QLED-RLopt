from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass(frozen=True)
class ParamDef:
    name: str
    low: float
    high: float


class ParameterSpace:
    """
    Research-grade QLED design parameter space.

    RL operates in normalized space: x in [-1, 1]^D.
    This class maps x <-> physical parameter dict and reports constraint violations.
    """

    def __init__(self):
        # ---- Research/industry parameters (v1) ----
        self.params: List[ParamDef] = [
            # thicknesses (nm)
            ParamDef("t_HTL_nm", 10.0, 60.0),
            ParamDef("t_EML_nm", 10.0, 40.0),
            ParamDef("t_ETL_nm", 10.0, 60.0),

            # injection / energy-level proxies (eV)
            ParamDef("phi_HTL_eV", 4.8, 5.6),
            ParamDef("phi_ETL_eV", 3.8, 4.5),

            # doping / defect proxies (0~1)
            ParamDef("p_doping_HTL", 0.0, 0.30),
            ParamDef("n_doping_ETL", 0.0, 0.30),

            # microstructure: PS microspheres + QD superlattice
            ParamDef("ps_radius_nm", 50.0, 250.0),
            ParamDef("ps_fill_frac", 0.00, 0.50),
            ParamDef("sl_thickness_nm", 8.0, 30.0),
            ParamDef("sl_gap_um", 0.3, 5.0),
            ParamDef("qd_coverage", 0.2, 1.0),

            # driving condition
            ParamDef("V_drive", 2.0, 6.0),
        ]

        self.names = [p.name for p in self.params]
        self.dim = len(self.params)

        # Keep 0 for now (we won't append metrics to obs yet)
        self.metrics_dim = 0

    # ---- sampling ----
    def sample_normalized(self, rng: np.random.Generator) -> np.ndarray:
        """Sample x in [-1, 1]^dim uniformly."""
        return rng.uniform(-1.0, 1.0, size=(self.dim,)).astype(np.float32)

    # ---- mapping: normalized <-> real ----
    def to_real(self, x_norm: np.ndarray) -> Dict[str, float]:
        """Map normalized vector x in [-1, 1]^D to physical parameter dict."""
        x = np.asarray(x_norm, dtype=np.float64)
        x = np.clip(x, -1.0, 1.0)

        out: Dict[str, float] = {}
        for i, p in enumerate(self.params):
            u = (x[i] + 1.0) / 2.0  # [-1,1] -> [0,1]
            val = p.low + u * (p.high - p.low)
            out[p.name] = float(val)
        return out

    def to_normalized(self, params_dict: Dict[str, float]) -> np.ndarray:
        """Map physical parameter dict to normalized vector x in [-1, 1]^D."""
        x = np.zeros((self.dim,), dtype=np.float32)
        for i, p in enumerate(self.params):
            v = float(params_dict[p.name])  # strict: must exist
            v = min(max(v, p.low), p.high)
            u = (v - p.low) / (p.high - p.low + 1e-12)  # [0,1]
            x[i] = float(2.0 * u - 1.0)  # [-1,1]
        return x

    # ---- constraints ----
    def constraint_violation(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Return constraint violations (>=0 means violated).
        We keep it simple but realistic; you can extend without breaking the interface.
        """
        v: Dict[str, float] = {}

        t_total = params["t_HTL_nm"] + params["t_EML_nm"] + params["t_ETL_nm"]
        v["t_total_over_180nm"] = max(0.0, t_total - 180.0)

        v["ps_fill_frac_over_0p45"] = max(0.0, params["ps_fill_frac"] - 0.45)

        v["sl_gap_under_0p5um"] = max(0.0, 0.5 - params["sl_gap_um"])

        v["V_drive_over_5p5V"] = max(0.0, params["V_drive"] - 5.5)

        return v

    def is_hard_invalid(self, violation: Any) -> bool:
        """Terminate early if violations are extreme."""
        if isinstance(violation, dict):
            return any(float(val) > 20.0 for val in violation.values())
        return float(violation) > 20.0

    # ---- metrics vectorization (optional) ----
    def metrics_to_vec(self, metrics: Dict[str, Any]) -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)
