from __future__ import annotations
import numpy as np
from .simulator_interface import SimulatorInterface


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class SurrogateSim(SimulatorInterface):
    """
    Physics-inspired surrogate for QLED metrics.
    Produces metrics in reasonable scale so RL can learn.

    Output metrics:
      - EQE: ~0..20 (percent-like)
      - recomb_overlap: 0..1
      - penalty: >=0
    """

    def evaluate(self, params):
        # ---- layer thickness effects (simple, interpretable) ----
        t_htl = params["t_HTL_nm"]
        t_eml = params["t_EML_nm"]
        t_etl = params["t_ETL_nm"]

        # ---- injection balance proxies ----
        # smaller barrier difference -> better balance
        phi_h = params["phi_HTL_eV"]
        phi_e = params["phi_ETL_eV"]
        barrier_mismatch = abs((phi_h - 5.2) - (4.1 - phi_e))  # proxy
        inj_balance = np.exp(- (barrier_mismatch ** 2) / 0.05)  # 0..1

        # doping improves injection but too much increases leakage
        p_d = params["p_doping_HTL"]
        n_d = params["n_doping_ETL"]
        inj_boost = _sigmoid(8.0 * (p_d + n_d - 0.15))  # 0..1

        # ---- recombination overlap proxy ----
        # too thick EML -> broad recomb, too thin -> pinhole risk
        overlap = np.exp(-((t_eml - 18.0) ** 2) / (2 * 7.0**2))  # peak ~18nm

        # ---- outcoupling gain proxy (your PS + superlattice core) ----
        ps_r = params["ps_radius_nm"]
        ps_ff = params["ps_fill_frac"]
        sl_t = params["sl_thickness_nm"]
        sl_gap = params["sl_gap_um"]
        qd_cov = params["qd_coverage"]

        # Mie-ish: radius near ~120nm is sweet spot (toy peak)
        mie_gain = 1.0 + 0.6 * np.exp(-((ps_r - 120.0) ** 2) / (2 * 50.0**2))
        # coverage: too low no effect, too high shorts/roughness
        cov_gain = 1.0 + 0.8 * np.clip(ps_ff / 0.30, 0.0, 1.0)
        # superlattice thickness: moderate is best
        sl_gain = 1.0 + 0.4 * np.exp(-((sl_t - 15.0) ** 2) / (2 * 6.0**2))
        # spacing: too tight hurts uniformity, too sparse loses benefit
        gap_gain = 1.0 + 0.3 * np.exp(-((sl_gap - 1.2) ** 2) / (2 * 0.8**2))
        # qd coverage affects radiative probability
        qd_gain = 0.6 + 0.4 * qd_cov  # 0.68..1.0

        outcoupling = mie_gain * cov_gain * sl_gain * gap_gain * qd_gain

        # ---- driving condition proxy ----
        V = params["V_drive"]
        # too high V => efficiency droop / heating penalty
        droop = np.exp(-max(0.0, V - 4.0) / 1.0)  # 1 at <=4V then decays

        # ---- penalties (short/leak/manufacturability) ----
        penalty = 0.0
        if ps_ff > 0.45:
            penalty += (ps_ff - 0.45) * 5.0
        if t_eml < 12.0:
            penalty += (12.0 - t_eml) * 0.2
        if sl_gap < 0.5:
            penalty += (0.5 - sl_gap) * 2.0

        # ---- final EQE (percent-like scale) ----
        # combine: injection * overlap * outcoupling * droop, scaled
        eqe = 20.0 * inj_balance * (0.5 + 0.5 * inj_boost) * overlap * outcoupling * droop
        eqe = float(max(0.0, eqe))

        return {
            "EQE": eqe,
            "recomb_overlap": float(overlap),
            "penalty": float(penalty),
            "inj_balance": float(inj_balance),
            "outcoupling": float(outcoupling),
            "droop": float(droop),
        }
