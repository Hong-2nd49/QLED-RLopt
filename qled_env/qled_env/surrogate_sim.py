from __future__ import annotations
import numpy as np
from .simulator_interface import SimulatorInterface


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class SurrogateSim(SimulatorInterface):
    """
    SurrogateSim v2: adds realistic-ish trade-offs.
    - brightness increases with V_drive (saturating)
    - droop decreases with brightness and V_drive
    - leakage includes baseline + doping term (can't be zero just because balance=1)
    - auger grows with brightness and high drive
    - lifetime decreases with leakage/auger/drive/doping
    """

    def evaluate(self, params):
        t_htl = float(params["t_HTL_nm"])
        t_eml = float(params["t_EML_nm"])
        t_etl = float(params["t_ETL_nm"])

        phi_h = float(params["phi_HTL_eV"])
        phi_e = float(params["phi_ETL_eV"])

        p_d = float(params["p_doping_HTL"])
        n_d = float(params["n_doping_ETL"])

        ps_r = float(params["ps_radius_nm"])
        ps_ff = float(params["ps_fill_frac"])
        sl_t = float(params["sl_thickness_nm"])
        sl_gap = float(params["sl_gap_um"])
        qd_cov = float(params["qd_coverage"])

        V = float(params["V_drive"])

        # --- injection balance proxy ---
        barrier_mismatch = abs((phi_h - 5.2) - (4.1 - phi_e))
        inj_balance = float(np.exp(- (barrier_mismatch ** 2) / 0.06))
        inj_boost = _sigmoid(8.0 * (p_d + n_d - 0.15))

        # --- recombination overlap peaks near ~18nm ---
        overlap = float(np.exp(-((t_eml - 18.0) ** 2) / (2 * 7.0**2)))

        # --- outcoupling proxies (PS + SL) ---
        mie_gain = 1.0 + 0.6 * np.exp(-((ps_r - 120.0) ** 2) / (2 * 55.0**2))
        cov_gain = 1.0 + 0.7 * np.clip(ps_ff / 0.30, 0.0, 1.0)
        sl_gain = 1.0 + 0.4 * np.exp(-((sl_t - 15.0) ** 2) / (2 * 6.0**2))
        gap_gain = 1.0 + 0.3 * np.exp(-((sl_gap - 1.2) ** 2) / (2 * 0.8**2))
        qd_gain = 0.6 + 0.4 * qd_cov
        outcoupling = float(mie_gain * cov_gain * sl_gain * gap_gain * qd_gain)

        # --- brightness driven by V (saturating), boosted by overlap & injection ---
        drive = max(0.0, V - 2.2)
        V_term = 1.0 - np.exp(-drive / 1.1)  # 0..~1
        brightness = float(250.0 + 1600.0 * V_term * (0.55 + 0.45 * inj_boost) * (0.55 + 0.45 * overlap))

        # --- leakage: baseline + doping + morphology + imbalance (not zero at balance=1) ---
        leakage = (
            0.05
            + 0.55 * (p_d + n_d)                # doping always adds some leakage
            + 0.45 * (1.0 - inj_balance)        # imbalance increases leakage
            + 0.30 * max(0.0, ps_ff - 0.30)      # too dense PS can create leakage paths
        )
        leakage = float(max(0.0, leakage))

        # --- auger & droop: worsen with brightness and high drive ---
        auger_rate = float((brightness / 1400.0) ** 2 * max(0.0, V - 3.2))
        droop = float(1.0 / (1.0 + (brightness / 1600.0) ** 2 + 0.6 * auger_rate))
        droop = _clamp(droop, 0.05, 1.0)

        # --- penalties (soft constraints) ---
        penalty = 0.0
        if ps_ff > 0.45:
            penalty += (ps_ff - 0.45) * 6.0
        if t_eml < 12.0:
            penalty += (12.0 - t_eml) * 0.25
        if sl_gap < 0.5:
            penalty += (0.5 - sl_gap) * 2.5

        # --- EQE proxy ---
        eqe_raw = 22.0 * inj_balance * (0.55 + 0.45 * inj_boost) * overlap * outcoupling * droop
        # soften extreme values (avoid crazy blow-up)
        eqe = float(40.0 * np.tanh(eqe_raw / 40.0))

        # --- lifetime proxy ---
        lifetime = 2000.0 / (1.0 + 2.5 * leakage + 3.5 * auger_rate + 0.20 * (max(0.0, V - 3.0) ** 2) + 0.8 * (p_d + n_d))
        lifetime = float(_clamp(lifetime, 50.0, 2000.0))

        return {
            "EQE": eqe,
            "recomb_overlap": overlap,
            "penalty": float(penalty),
            "inj_balance": inj_balance,
            "outcoupling": outcoupling,
            "droop": droop,
            "brightness": brightness,
            "leakage": leakage,
            "auger_rate": auger_rate,
            "lifetime": lifetime,
        }
