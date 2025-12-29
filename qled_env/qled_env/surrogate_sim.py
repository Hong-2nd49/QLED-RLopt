from __future__ import annotations
import numpy as np
from .simulator_interface import SimulatorInterface


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class SurrogateSim(SimulatorInterface):
    """
    Physics-inspired surrogate for QLED metrics.
    Outputs additional proxies for analysis:
      brightness, leakage, auger_rate, lifetime
    """

    def evaluate(self, params):
        t_htl = params["t_HTL_nm"]
        t_eml = params["t_EML_nm"]
        t_etl = params["t_ETL_nm"]

        phi_h = params["phi_HTL_eV"]
        phi_e = params["phi_ETL_eV"]
        barrier_mismatch = abs((phi_h - 5.2) - (4.1 - phi_e))
        inj_balance = np.exp(- (barrier_mismatch ** 2) / 0.05)

        p_d = params["p_doping_HTL"]
        n_d = params["n_doping_ETL"]
        inj_boost = _sigmoid(8.0 * (p_d + n_d - 0.15))

        overlap = np.exp(-((t_eml - 18.0) ** 2) / (2 * 7.0**2))

        ps_r = params["ps_radius_nm"]
        ps_ff = params["ps_fill_frac"]
        sl_t = params["sl_thickness_nm"]
        sl_gap = params["sl_gap_um"]
        qd_cov = params["qd_coverage"]

        mie_gain = 1.0 + 0.6 * np.exp(-((ps_r - 120.0) ** 2) / (2 * 50.0**2))
        cov_gain = 1.0 + 0.8 * np.clip(ps_ff / 0.30, 0.0, 1.0)
        sl_gain = 1.0 + 0.4 * np.exp(-((sl_t - 15.0) ** 2) / (2 * 6.0**2))
        gap_gain = 1.0 + 0.3 * np.exp(-((sl_gap - 1.2) ** 2) / (2 * 0.8**2))
        qd_gain = 0.6 + 0.4 * qd_cov

        outcoupling = mie_gain * cov_gain * sl_gain * gap_gain * qd_gain

        V = params["V_drive"]
        droop = np.exp(-max(0.0, V - 4.0) / 1.0)

        penalty = 0.0
        if ps_ff > 0.45:
            penalty += (ps_ff - 0.45) * 5.0
        if t_eml < 12.0:
            penalty += (12.0 - t_eml) * 0.2
        if sl_gap < 0.5:
            penalty += (0.5 - sl_gap) * 2.0

        eqe = 20.0 * inj_balance * (0.5 + 0.5 * inj_boost) * overlap * outcoupling * droop
        eqe = float(max(0.0, eqe))

        # --- extra proxies (optional for research plots) ---
        brightness = 1000.0 * (0.3 + 0.7 * inj_boost) * (0.5 + 0.5 * overlap) * droop
        leakage = max(0.0, (1.0 - inj_balance)) * (0.5 + p_d + n_d) * (1.0 + max(0.0, ps_ff - 0.30))
        auger_rate = (max(0.0, V - 4.0) ** 2) * (0.5 + 0.5 * inj_boost) * (0.5 + 0.5 * brightness / 1000.0)
        lifetime = 2000.0 / (1.0 + 3.0 * leakage + 5.0 * auger_rate + max(0.0, V - 4.0))

        return {
            "EQE": eqe,
            "recomb_overlap": float(overlap),
            "penalty": float(penalty),
            "inj_balance": float(inj_balance),
            "outcoupling": float(outcoupling),
            "droop": float(droop),
            "brightness": float(brightness),
            "leakage": float(leakage),
            "auger_rate": float(auger_rate),
            "lifetime": float(lifetime),
        }
