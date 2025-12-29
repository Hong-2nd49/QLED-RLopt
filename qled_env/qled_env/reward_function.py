from __future__ import annotations
from typing import Dict, Any, Tuple
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _log1p_norm(x: float, ref: float) -> float:
    x = max(0.0, float(x))
    ref = max(1e-12, float(ref))
    return _clamp(math.log1p(x) / math.log1p(ref), 0.0, 2.0)


def compute_reward(
    metrics: Dict[str, Any],
    violation: Any = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    # ---- core metrics ----
    eqe = _safe_float(metrics.get("EQE", metrics.get("eqe", 0.0)), 0.0)
    overlap = _clamp(_safe_float(metrics.get("recomb_overlap", 0.0), 0.0), 0.0, 1.0)
    balance = _clamp(_safe_float(metrics.get("inj_balance", 0.0), 0.0), 0.0, 1.0)
    droop = _clamp(_safe_float(metrics.get("droop", 1.0), 1.0), 0.0, 1.0)

    metric_penalty = _safe_float(metrics.get("penalty", 0.0), 0.0)
    U_prev = metrics.get("U_prev", None)
    U_prev = None if U_prev is None else _safe_float(U_prev, None)
    delta_params_norm = _safe_float(metrics.get("delta_params_norm", 0.0), 0.0)

    # optional metrics
    brightness = _safe_float(metrics.get("brightness", 0.0), 0.0)
    lifetime = _safe_float(metrics.get("lifetime", 0.0), 0.0)
    auger = _safe_float(metrics.get("auger_rate", metrics.get("auger", 0.0)), 0.0)
    leakage = _safe_float(metrics.get("leakage", 0.0), 0.0)

    # ---- constraint penalties ----
    if isinstance(violation, dict):
        constraint_penalty = sum(max(0.0, _safe_float(v, 0.0)) for v in violation.values())
    else:
        constraint_penalty = max(0.0, _safe_float(violation, 0.0))

    total_penalty = metric_penalty + constraint_penalty

    # ---- shaped components ----
    eqe_s = _log1p_norm(eqe, ref=25.0)  # allow >20% without instantly saturating
    overlap_s = overlap ** 2
    balance_s = balance ** 2

    # droop already in SurrogateSim's EQE; keep tiny preference only
    droop_s = droop ** 2

    bri_s = _log1p_norm(brightness, ref=2000.0)
    life_s = _log1p_norm(lifetime, ref=2000.0)
    auger_s = _log1p_norm(auger, ref=1.0)
    leak_s = _log1p_norm(leakage, ref=1.0)

    # ---- weights (less aggressive to avoid clip saturation) ----
    w_eqe = 1.6
    w_overlap = 0.9
    w_balance = 0.9
    w_droop = 0.05

    w_bri = 0.08
    w_life = 0.12
    w_auger = 0.18
    w_leak = 0.18

    # penalties
    w_pen = 2.2          # stronger penalty effect
    w_jump = 0.8         # discourage big jumps
    w_delta = 0.25       # small progress shaping

    # penalty shaping: soft but meaningful
    pen_term = math.log1p(max(0.0, total_penalty))

    # extra penalty if penalty exceeds a comfort threshold (soft hinge)
    pen_hinge = max(0.0, total_penalty - 0.10)
    pen_hinge_term = pen_hinge ** 2

    U = (
        w_eqe * eqe_s
        + w_overlap * overlap_s
        + w_balance * balance_s
        + w_droop * droop_s
        + w_bri * bri_s
        + w_life * life_s
        - w_auger * auger_s
        - w_leak * leak_s
        - w_pen * pen_term
        - 5.0 * pen_hinge_term
        - w_jump * (delta_params_norm ** 2)
    )

    dU = 0.0
    if U_prev is not None:
        dU = U - U_prev

    # tiny threshold bonus, won't dominate
    bonus = 0.0
    if eqe > 20.0:
        bonus += 0.08
    if eqe > 30.0:
        bonus += 0.05

    raw = U + w_delta * dU + bonus

    # ---- bounded reward avoids hard saturation at +10/-10 ----
    reward = 10.0 * math.tanh(raw)

    info = {
        "U": U,
        "dU": dU,
        "bonus": bonus,
        "raw": raw,
        "reward": reward,
        "eqe": eqe,
        "eqe_s": eqe_s,
        "overlap": overlap,
        "inj_balance": balance,
        "droop": droop,
        "total_penalty": total_penalty,
        "pen_term": pen_term,
        "pen_hinge_term": pen_hinge_term,
        "delta_params_norm": delta_params_norm,
    }
    return float(reward), info
